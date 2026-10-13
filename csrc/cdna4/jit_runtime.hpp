#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
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

struct JitModule {
    hipModule_t module = nullptr;
    hipFunction_t gemm = nullptr;
    hipFunction_t reduce = nullptr;

    ~JitModule() {
        if (module != nullptr)
            (void)hipModuleUnload(module);
    }
};

class JitRuntime {
public:
    void init(const std::string& root) {
        std::lock_guard<std::mutex> lock(mutex_);
        root_ = std::filesystem::absolute(root).string();
        include_root_ = std::filesystem::exists(std::filesystem::path(root_) / "include")
            ? (std::filesystem::path(root_) / "include").string()
            : (std::filesystem::path(root_) / "deep_gemm" / "include").string();
    }

    size_t workspace_size(const KernelConfig& config, const int m, const int n) const {
        if (config.variant != KernelVariant::mfma_16x16x128_split_k || config.split_k <= 1)
            return 0;
        const size_t row = static_cast<size_t>((n + 3) / 4 * 4) * sizeof(float);
        return static_cast<size_t>(config.split_k) * static_cast<size_t>(m) * row;
    }

    // tiles: output tiles of the launch (normal GEMMs: 0, derived from the shape). Persistent
    // kernels take at most one workgroup per compute unit.
    int launch(GemmArgs& args, const KernelConfig& config, const int out_dtype,
               hipStream_t stream, const int num_cus, int tiles = 0) {
        const auto jit = get_module(config, out_dtype);
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

        void* params[] = {&args};
        auto error = hipModuleLaunchKernel(jit->gemm, grid, 1, 1, block, 1, 1, shared, stream, params, nullptr);
        if (error != hipSuccess)
            return static_cast<int>(error);
        if (config.variant == KernelVariant::mfma_16x16x128_split_k) {
            const unsigned int bx = (args.N + 1023) / 1024;
            const unsigned int by = std::min(args.M, std::max(1, 4096 / static_cast<int>(std::max(bx, 1u))));
            void* reduce_params[] = {&args};
            error = hipModuleLaunchKernel(jit->reduce, bx, by, 1, 256, 1, 1, 0, stream, reduce_params, nullptr);
        }
        return static_cast<int>(error);
    }

private:
    static std::pair<int, std::string> device_arch() {
        int device = 0;
        (void)hipGetDevice(&device);
        hipDeviceProp_t properties{};
        (void)hipGetDeviceProperties(&properties, device);
        return {device, properties.gcnArchName};
    }

    static const char* variant_name(const KernelVariant variant) {
        switch (variant) {
        case KernelVariant::mfma_16x16x128_data_parallel: return "mfma_16x16x128_data_parallel";
        case KernelVariant::mfma_16x16x128_persistent: return "mfma_16x16x128_persistent";
        case KernelVariant::mfma_16x16x128_split_k: return "mfma_16x16x128_split_k";
        case KernelVariant::mfma_16x16x128_small_m: return "mfma_16x16x128_small_m";
        }
        return "unknown";
    }

    static unsigned int shared_bytes(const KernelConfig& config) {
        if (config.variant == KernelVariant::mfma_16x16x128_small_m) // a tile per wave, padded rows
            return config.waves * config.block_m * (config.block_n + 4) * sizeof(float);
        // MFMA 16x16x128 pipeline: two operand stages, plus the persistent BF16 epilogue buffer.
        const unsigned int stage = 256 * 128 + 256 * 128 + 256 * 4 + 256 * 4;
        const unsigned int epilogue = config.variant == KernelVariant::mfma_16x16x128_persistent
            ? 32 * (2 * 256 + 16) : 0;
        return 2 * stage + epilogue;
    }

    static const char* type_name(const GemmType type) {
        switch (type) {
        case GemmType::Normal: return "Normal";
        case GemmType::MGroupedContiguous: return "MGroupedContiguous";
        case GemmType::MGroupedContiguousPsum: return "MGroupedContiguousPsum";
        case GemmType::MGroupedMasked: return "MGroupedMasked";
        case GemmType::KGroupedContiguous: return "KGroupedContiguous";
        case GemmType::KGroupedContiguousPsum: return "KGroupedContiguousPsum";
        }
        return "unknown";
    }

    static const char* operand_name(const OperandType type) {
        switch (type) {
        case OperandType::Fp8: return "Fp8";
        case OperandType::Fp4: return "Fp4";
        case OperandType::Bf16: return "Bf16";
        }
        return "unknown";
    }

    static std::string operand_types(const KernelConfig& config) {
        return std::string("OperandType::") + operand_name(config.a_type) + ",OperandType::" +
               operand_name(config.b_type);
    }

    std::string source_for(const KernelConfig& config, const int out_dtype) const {
        const char* output = out_dtype == MXFP8_OUT_FP32 ? "float" : "std::uint16_t";
        if (config.variant == KernelVariant::mfma_16x16x128_small_m) {
            std::ostringstream source;
            const int threads = 64 * config.waves;
            source << "#include <deep_gemm/impls/cdna4/fp8_gemm_mfma_16x16x128_small_m.cuh>\n"
                   << "extern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(" << threads << ","
                   << threads << "))) void dg_cdna4_gemm(GemmArgs p) { "
                      "fp8_gemm_mfma_16x16x128_small_m_impl<Mfma16x16x128SmallMConfig<"
                   << config.block_m / 16 << "," << config.block_n / 16 << "," << config.depth << ","
                   << config.waves << "," << config.scale_gran_128 << ",GemmType::" << type_name(config.type) << ","
                   << (config.k_tail ? "true" : "false") << "," << operand_types(config) << ">," << output
                   << ">(p); }\n";
            return source.str();
        }
        const char* mode = config.variant == KernelVariant::mfma_16x16x128_persistent ? "Persistent" :
                           config.variant == KernelVariant::mfma_16x16x128_split_k ? "SplitK" : "DataParallel";
        std::ostringstream source;
        source << "#include <deep_gemm/impls/cdna4/fp8_gemm_mfma_16x16x128.cuh>\n";
        if (config.variant == KernelVariant::mfma_16x16x128_split_k)
            source << "#include <deep_gemm/impls/cdna4/split_k_reduce.cuh>\n";
        source << "extern \"C\" __global__ "
                  "__attribute__((amdgpu_flat_work_group_size(256,256), amdgpu_waves_per_eu(1,1))) "
                  "void dg_cdna4_gemm(GemmArgs p) { "
                  "fp8_gemm_mfma_16x16x128_impl<Mfma16x16x128GemmConfig<Mfma16x16x128GemmMode::"
               << mode << "," << config.mn_major << "," << config.scale_gran_128
               << ",GemmType::" << type_name(config.type) << "," << operand_types(config) << ">," << output
               << ">(p); }\n";
        if (config.variant == KernelVariant::mfma_16x16x128_split_k)
            source << "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_reduce(GemmArgs p) {"
                   << " splitk_reduce_impl<" << output << ">(p); }\n";
        return source.str();
    }

    std::shared_ptr<JitModule> get_module(const KernelConfig& config, const int out_dtype) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [device, arch] = device_arch();
        const std::string key = std::to_string(device) + ":" + arch + ":" +
                                variant_name(config.variant) + ":" + std::to_string(out_dtype) + ":" +
                                std::to_string(config.mn_major) + ":" + std::to_string(config.scale_gran_128) +
                                ":" + type_name(config.type) + ":" + std::to_string(config.block_m) + "x" +
                                std::to_string(config.block_n) + ":" + std::to_string(config.depth) + ":" +
                                std::to_string(config.waves) + (config.k_tail ? ":tail" : "") + ":" +
                                operand_name(config.a_type) + "x" + operand_name(config.b_type);
        auto found = modules_.find(key);
        if (found != modules_.end())
            return found->second;
        if (include_root_.empty())
            throw std::runtime_error("CDNA4 HIP JIT runtime was not initialized");

        const auto source = source_for(config, out_dtype);
        hiprtcProgram program = nullptr;
        auto status = hiprtcCreateProgram(&program, source.c_str(), "deep_gemm_cdna4.cpp", 0, nullptr, nullptr);
        if (status != HIPRTC_SUCCESS)
            throw std::runtime_error("hiprtcCreateProgram failed");
        const std::string include = "-I" + include_root_;
        const char* rocm = std::getenv("ROCM_PATH");
        if (rocm == nullptr) rocm = std::getenv("ROCM_HOME");
        if (rocm == nullptr) rocm = "/opt/rocm";
        const std::string rocm_include = "-I" + (std::filesystem::path(rocm) / "include").string();
        const std::string arch_option = "--offload-arch=" + arch;
        const char* options[] = {"--std=c++17", include.c_str(), rocm_include.c_str(), arch_option.c_str()};
        status = hiprtcCompileProgram(program, 4, options);
        if (status != HIPRTC_SUCCESS) {
            size_t log_size = 0;
            hiprtcGetProgramLogSize(program, &log_size);
            std::string log(log_size, '\0');
            if (log_size)
                hiprtcGetProgramLog(program, log.data());
            hiprtcDestroyProgram(&program);
            throw std::runtime_error("HIPRTC compilation failed: " + log);
        }
        size_t code_size = 0;
        hiprtcGetCodeSize(program, &code_size);
        std::vector<char> code(code_size);
        hiprtcGetCode(program, code.data());
        hiprtcDestroyProgram(&program);

        auto module = std::make_shared<JitModule>();
        if (hipModuleLoadData(&module->module, code.data()) != hipSuccess ||
            hipModuleGetFunction(&module->gemm, module->module, "dg_cdna4_gemm") != hipSuccess)
            throw std::runtime_error("loading HIPRTC code object failed");
        if (config.variant == KernelVariant::mfma_16x16x128_split_k &&
            hipModuleGetFunction(&module->reduce, module->module, "dg_cdna4_reduce") != hipSuccess)
            throw std::runtime_error("loading HIPRTC reduction kernel failed");
        modules_.emplace(key, module);
        return module;
    }

    std::mutex mutex_;
    std::string root_;
    std::string include_root_;
    std::unordered_map<std::string, std::shared_ptr<JitModule>> modules_;
};

inline JitRuntime jit_runtime;

}  // namespace deep_gemm::cdna4
