#pragma once

#include <deep_jit/backend/hip/backend.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cost_model.hpp"
#include "deep_gemm/impls/cdna4/gemm_args.h"

namespace deep_gemm::cdna4 {

using JitModule = deep_jit::hip::Kernel;
using HipJit = deep_jit::Runtime<deep_jit::HIP>;

class JitRuntime {
  public:
    void init(const std::string &root) {
        deep_jit::GilScopedRelease release;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto absolute = std::filesystem::absolute(root).string();
        if (!root_.empty()) {
            if (root_ != absolute)
                throw std::runtime_error("cannot change an initialized CDNA4 JIT root");
            return;
        }
        root_ = std::filesystem::absolute(root).string();
        include_root_ = std::filesystem::exists(std::filesystem::path(root_) / "include")
                            ? (std::filesystem::path(root_) / "include").string()
                            : (std::filesystem::path(root_) / "deep_gemm" / "include").string();
        // uni_symm device headers: copied into the package include root by the wheel build, in
        // third-party/ of a source checkout
        const auto checkout_symm =
            std::filesystem::path(include_root_) / ".." / ".." / "third-party" / "uni_symm" / "include";
        symm_include_root_ = !std::filesystem::exists(std::filesystem::path(include_root_) / "uni_symm") &&
                                     std::filesystem::exists(checkout_symm)
                                 ? std::filesystem::canonical(checkout_symm).string()
                                 : include_root_;
    }

    size_t workspace_size(const KernelConfig &config, const int m, const int n) const {
        if (config.variant != KernelVariant::mfma_16x16x128_split_k || config.split_k <= 1)
            return 0;
        const size_t row = static_cast<size_t>((n + 3) / 4 * 4) * sizeof(float);
        return static_cast<size_t>(config.split_k) * static_cast<size_t>(m) * row;
    }

    // tiles: output tiles of the launch (normal GEMMs: 0, derived from the shape). Persistent
    // kernels take at most one workgroup per compute unit.
    int launch(GemmArgs &args, const KernelConfig &config, const int out_dtype, hipStream_t stream, const int num_cus,
               int tiles = 0) {
        const auto jit = get_gemm_module(config, out_dtype);
        if (tiles == 0)
            tiles = ceil_div(args.M, config.block_m) * ceil_div(args.N, config.block_n);
        if (tiles == 0)
            return 0;
        unsigned int grid = tiles;
        unsigned int block = config.variant == KernelVariant::mfma_16x16x128_small_m ? 64 * config.waves : 256;
        unsigned int shared = shared_bytes(config);
        if (config.variant == KernelVariant::mfma_16x16x128_persistent)
            grid = std::min(tiles, std::max(num_cus, 1));
        else if (config.variant == KernelVariant::mfma_16x16x128_split_k)
            grid = tiles * std::max(config.split_k, 1);
        else if (config.variant != KernelVariant::mfma_16x16x128_small_m &&
                 (config.type == GemmType::MGroupedContiguousPsum || config.type == GemmType::MGroupedMasked))
            // device-counted tiles: one round of resident workgroups walks them
            grid = std::min(tiles, std::max(num_cus, 1));

        hipError_t error;
        if (config.pipeline_combine) {
            grid = num_cus; // Reserve the communication team even for a single useful tile.
            int occupancy = 0;
            if (hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, jit->kernels[0], block, shared) !=
                    hipSuccess ||
                occupancy < 1)
                throw std::runtime_error("pipelined combine needs a resident workgroup per CU");
            error = JitModule::launch_function(jit->kernels[0], dim3(grid), dim3(block), shared, stream, true, args);
        } else
            error = JitModule::launch_function(jit->kernels[0], dim3(grid), dim3(block), shared, stream, false, args);
        if (error != hipSuccess)
            return static_cast<int>(error);
        if (config.variant == KernelVariant::mfma_16x16x128_split_k) {
            const unsigned int bx = (args.N + 1023) / 1024;
            const unsigned int by = std::min(args.M, std::max(1, 4096 / static_cast<int>(std::max(bx, 1u))));
            error = JitModule::launch_function(jit->kernels[1], dim3(bx, by), dim3(256), 0, stream, false, args);
        }
        return static_cast<int>(error);
    }

    // A module compiled from source() on the first use of key (per device and architecture), with
    // the kernels `names`; options are added to the compiler's
    std::shared_ptr<JitModule> get_module(const std::string &key, const std::function<std::string()> &source,
                                          const std::vector<std::string> &names,
                                          const std::vector<std::string> &options = {}) {
        // Release Python's GIL before waiting for the mutex: compilation also
        // releases it, so acquiring these locks in the opposite order deadlocks.
        deep_jit::GilScopedRelease release;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [device, arch] = device_arch();
        deep_jit::hash::FNV1a signature;
        signature.update(key);
        signature.update("kernel_names");
        signature.update(std::to_string(names.size()));
        for (const auto &name : names)
            signature.update(name);
        signature.update("compiler_options");
        signature.update(std::to_string(options.size()));
        for (const auto &option : options)
            signature.update(option);
        const std::string full_key = std::to_string(device) + ":" + signature.get_hex_digest();
        if (auto found = modules_.find(full_key); found != modules_.end())
            return found->second;
        auto runtime = runtime_for_device(device);
        deep_jit::hip::CompilerOptions compiler_options{.kernel_names = names, .extra_hiprtc_flags = options};
        auto module = runtime->compile("cdna4", source(), compiler_options);
        modules_.emplace(full_key, module);
        return module;
    }

    std::shared_ptr<HipJit> get_runtime() {
        deep_jit::GilScopedRelease release;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [device, arch] = device_arch();
        return runtime_for_device(device);
    }

    // Launch a kernel taking one argument struct by value
    template <class Args>
    static int launch_kernel(hipFunction_t kernel, Args &args, dim3 grid, dim3 block, unsigned int shared,
                             hipStream_t stream) {
        return static_cast<int>(JitModule::launch_function(kernel, grid, block, shared, stream, false, args));
    }

  private:
    static std::pair<int, std::string> device_arch() {
        int device = 0;
        DJ_HIP_CHECK(hipGetDevice(&device));
        hipDeviceProp_t properties{};
        DJ_HIP_CHECK(hipGetDeviceProperties(&properties, device));
        return {device, properties.gcnArchName};
    }

    static const char *variant_name(const KernelVariant variant) {
        switch (variant) {
        case KernelVariant::mfma_16x16x128_data_parallel:
            return "mfma_16x16x128_data_parallel";
        case KernelVariant::mfma_16x16x128_persistent:
            return "mfma_16x16x128_persistent";
        case KernelVariant::mfma_16x16x128_split_k:
            return "mfma_16x16x128_split_k";
        case KernelVariant::mfma_16x16x128_small_m:
            return "mfma_16x16x128_small_m";
        }
        return "unknown";
    }

    static unsigned int shared_bytes(const KernelConfig &config) {
        if (config.variant == KernelVariant::mfma_16x16x128_small_m) // a tile per wave, padded rows
            return config.waves * config.block_m * (config.block_n + 4) * sizeof(float);
        // MFMA 16x16x128 pipeline: two operand stages, plus the persistent BF16 epilogue buffer.
        const unsigned int stage = 256 * 128 + 256 * 128 + 256 * 4 + 256 * 4;
        const unsigned int epilogue =
            config.variant == KernelVariant::mfma_16x16x128_persistent ? 32 * (2 * 256 + 16) : 0;
        return 2 * stage + epilogue;
    }

    static const char *type_name(const GemmType type) {
        switch (type) {
        case GemmType::Normal:
            return "Normal";
        case GemmType::MGroupedContiguous:
            return "MGroupedContiguous";
        case GemmType::MGroupedContiguousPsum:
            return "MGroupedContiguousPsum";
        case GemmType::MGroupedMasked:
            return "MGroupedMasked";
        case GemmType::KGroupedContiguous:
            return "KGroupedContiguous";
        case GemmType::KGroupedContiguousPsum:
            return "KGroupedContiguousPsum";
        case GemmType::KBatched:
            return "KBatched";
        }
        return "unknown";
    }

    static const char *operand_name(const OperandType type) {
        switch (type) {
        case OperandType::Fp8:
            return "Fp8";
        case OperandType::Fp4:
            return "Fp4";
        case OperandType::Bf16:
            return "Bf16";
        }
        return "unknown";
    }

    static std::string operand_types(const KernelConfig &config) {
        return std::string("OperandType::") + operand_name(config.a_type) +
               ",OperandType::" + operand_name(config.b_type);
    }

    std::string source_for(const KernelConfig &config, const int out_dtype) const {
        const char *output = out_dtype == MXFP8_OUT_FP32 ? "float" : "std::uint16_t";
        if (config.variant == KernelVariant::mfma_16x16x128_small_m) {
            std::ostringstream source;
            const int threads = 64 * config.waves;
            source << "#include <deep_gemm/impls/cdna4/fp8_gemm_mfma_16x16x128_small_m.cuh>\n"
                   << "extern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(" << threads << "," << threads
                   << "))) void dg_cdna4_gemm(GemmArgs p) { "
                      "fp8_gemm_mfma_16x16x128_small_m_impl<Mfma16x16x128SmallMConfig<"
                   << config.block_m / 16 << "," << config.block_n / 16 << "," << config.depth << "," << config.waves
                   << "," << config.scale_gran_128 << ",GemmType::" << type_name(config.type) << ","
                   << (config.k_tail ? "true" : "false") << "," << operand_types(config) << ">," << output
                   << ">(p); }\n";
            return source.str();
        }
        const char *mode = config.variant == KernelVariant::mfma_16x16x128_persistent ? "Persistent"
                           : config.variant == KernelVariant::mfma_16x16x128_split_k  ? "SplitK"
                                                                                      : "DataParallel";
        std::ostringstream source;
        source << "#include <deep_gemm/impls/cdna4/fp8_gemm_mfma_16x16x128.cuh>\n";
        if (config.variant == KernelVariant::mfma_16x16x128_split_k)
            source << "#include <deep_gemm/impls/cdna4/split_k_reduce.cuh>\n";
        source << "extern \"C\" __global__ "
                  "__attribute__((amdgpu_flat_work_group_size(256,256), amdgpu_waves_per_eu(1,1))) "
                  "void dg_cdna4_gemm(GemmArgs p) { "
                  "fp8_gemm_mfma_16x16x128_impl<Mfma16x16x128GemmConfig<Mfma16x16x128GemmMode::"
               << mode << "," << config.mn_major << "," << config.scale_gran_128
               << ",GemmType::" << type_name(config.type) << "," << operand_types(config) << ","
               << int(config.fused_swiglu) << ","
               << (config.pipeline_combine ? (config.pipeline_profile ? 3 : 2) : int(config.direct_combine)) << ","
               << config.tile_group_m << "," << config.copy_cache_policy << "," << config.publish_cache_policy << ","
               << int(config.moe_special_epilogue) << ">," << output << ">(p); }\n";
        if (config.variant == KernelVariant::mfma_16x16x128_split_k)
            source << "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_reduce(GemmArgs p) {"
                   << " splitk_reduce_impl<" << output << ">(p); }\n";
        return source.str();
    }

    std::shared_ptr<JitModule> get_gemm_module(const KernelConfig &config, const int out_dtype) {
        const std::string key =
            std::string("gemm:") + variant_name(config.variant) + ":" + std::to_string(out_dtype) + ":" +
            std::to_string(config.mn_major) + ":" + std::to_string(config.scale_gran_128) + ":" +
            type_name(config.type) + ":" + std::to_string(config.block_m) + "x" + std::to_string(config.block_n) + ":" +
            std::to_string(config.depth) + ":" + std::to_string(config.waves) + (config.k_tail ? ":tail" : "") + ":" +
            operand_name(config.a_type) + "x" + operand_name(config.b_type) + (config.fused_swiglu ? ":swiglu" : "") +
            (config.pipeline_combine ? (config.pipeline_profile ? ":pipeline-profile" : ":pipeline-combine")
             : config.direct_combine ? ":combine"
                                     : "") +
            ":gm" + std::to_string(config.tile_group_m) + ":copy-cache" + std::to_string(config.copy_cache_policy) +
            ":publish-cache" + std::to_string(config.publish_cache_policy) +
            (config.moe_special_epilogue ? ":special-epi" + std::to_string(config.moe_special_epilogue) : "");
        std::vector<std::string> names = {"dg_cdna4_gemm"};
        if (config.variant == KernelVariant::mfma_16x16x128_split_k)
            names.push_back("dg_cdna4_reduce");
        return get_module(key, [&] { return source_for(config, out_dtype); }, names);
    }

    std::shared_ptr<HipJit> runtime_for_device(int device) {
        auto &runtime = runtimes_[device];
        if (!runtime) {
            if (include_root_.empty())
                throw std::runtime_error("CDNA4 JIT runtime was not initialized");
            runtime = std::make_shared<HipJit>(deep_jit::Config(
                root_, "DG", "cdna4-hip-v1", {include_root_, symm_include_root_}, {"deep_gemm/", "uni_symm/"}));
            // Keep the existing device compiler mode and numerical semantics.
            runtime->default_compiler_options.cpp_standard = 17;
        }
        return runtime;
    }

    std::mutex mutex_;
    std::string root_;
    std::string include_root_, symm_include_root_;
    std::unordered_map<std::string, std::shared_ptr<JitModule>> modules_;
    std::unordered_map<int, std::shared_ptr<HipJit>> runtimes_;
};

inline JitRuntime jit_runtime;

} // namespace deep_gemm::cdna4
