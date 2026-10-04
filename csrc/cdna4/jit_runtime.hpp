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
        if (config.variant != KernelVariant::split_k || config.split_k <= 1)
            return 0;
        const size_t row = static_cast<size_t>((n + 3) / 4 * 4) * sizeof(float);
        return static_cast<size_t>(config.split_k) * static_cast<size_t>(m) * row;
    }

    int launch(GemmArgs& args, const KernelConfig& config, const int out_dtype,
               hipStream_t stream, const int num_cus) {
        const auto jit = get_module(config, out_dtype);
        const int tile_m = config.block_m;
        const int tile_n = config.block_n;
        const int tiles = (args.M + tile_m - 1) / tile_m * (args.N + tile_n - 1) / tile_n;
        unsigned int grid = tiles;
        unsigned int block = 256;
        unsigned int shared = shared_bytes(config);
        if (config.variant == KernelVariant::persistent)
            grid = std::min(tiles, std::max(num_cus, 1));
        else if (config.variant == KernelVariant::split_k)
            grid = tiles * std::max(config.split_k, 1);
        else if (config.variant == KernelVariant::tiled_128)
            block = 256;
        else if (config.variant == KernelVariant::tiled_256)
            block = 1024;

        void* params[] = {&args};
        auto error = hipModuleLaunchKernel(jit->gemm, grid, 1, 1, block, 1, 1, shared, stream, params, nullptr);
        if (error != hipSuccess)
            return static_cast<int>(error);
        if (config.variant == KernelVariant::split_k) {
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
        case KernelVariant::tiled_128: return "tiled128";
        case KernelVariant::tiled_256: return "tiled256";
        case KernelVariant::data_parallel: return "data";
        case KernelVariant::persistent: return "persistent";
        case KernelVariant::split_k: return "splitk";
        }
        return "unknown";
    }

    static unsigned int shared_bytes(const KernelConfig& config) {
        if (config.variant == KernelVariant::tiled_128)
            return 2 * (128 * 128 * 2);
        if (config.variant == KernelVariant::tiled_256)
            return 256 * 256 * 2;
        // FP8 pipeline: two operand stages, plus the persistent BF16 epilogue buffer.
        const unsigned int stage = 256 * 128 + 256 * 128 + 256 * 4 + 256 * 4;
        const unsigned int epilogue = config.variant == KernelVariant::persistent ? 32 * (2 * 256 + 16) : 0;
        return 2 * stage + epilogue;
    }

    std::string source_for(const KernelConfig& config, const int out_dtype) const {
        const char* output = out_dtype == MXFP8_OUT_FP32 ? "float" : "std::uint16_t";
        std::ostringstream source;
        source << "#include <cstdint>\n";
        const bool tiled = config.variant == KernelVariant::tiled_128 ||
                           config.variant == KernelVariant::tiled_256;
        source << (tiled ? "#include <deep_gemm/impls/cdna4/gemm_tiled.h>\n"
                         : "#include <deep_gemm/impls/cdna4/fp8_gemm.cuh>\n");
        if (config.variant == KernelVariant::split_k)
            source << "#include <deep_gemm/impls/cdna4/split_k_reduce.cuh>\n";
        if (config.variant == KernelVariant::tiled_128) {
            source << "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_gemm(GemmArgs p) {"
                   << " gemm_tiled_impl<Tiled128x128S2, " << output << ">(p); }\n";
        } else if (config.variant == KernelVariant::tiled_256) {
            source << "extern \"C\" __global__ __launch_bounds__(1024) void dg_cdna4_gemm(GemmArgs p) {"
                   << " gemm_tiled_impl<Tiled256x256S1, " << output << ">(p); }\n";
        } else if (config.variant == KernelVariant::persistent) {
            source << "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_gemm(GemmArgs p) {"
                   << " fp8_gemm_impl<Fp8GemmConfig<Fp8GemmMode::Persistent>, " << output << ">(p); }\n";
        } else if (config.variant == KernelVariant::split_k) {
            source << "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_gemm(GemmArgs p) {"
                   << " fp8_gemm_impl<Fp8GemmConfig<Fp8GemmMode::SplitK>, " << output << ">(p); }\n";
            source << "extern \"C\" __global__ void dg_cdna4_reduce(GemmArgs p) {"
                   << " splitk_reduce_impl<" << output << ">(p); }\n";
        } else {
            source << "extern \"C\" __global__ __launch_bounds__(256) void dg_cdna4_gemm(GemmArgs p) {"
                   << " fp8_gemm_impl<Fp8GemmConfig<Fp8GemmMode::DataParallel>, " << output << ">(p); }\n";
        }
        return source.str();
    }

    std::shared_ptr<JitModule> get_module(const KernelConfig& config, const int out_dtype) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [device, arch] = device_arch();
        const std::string key = std::to_string(device) + ":" + arch + ":" +
                                variant_name(config.variant) + ":" + std::to_string(out_dtype);
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
        if (config.variant == KernelVariant::split_k &&
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
