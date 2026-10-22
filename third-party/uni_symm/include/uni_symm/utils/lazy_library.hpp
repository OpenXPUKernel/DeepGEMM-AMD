#pragma once

#include <dlfcn.h>

#include <mutex>
#include <string>

#include <uni_symm/utils/exception.hpp>

namespace uni_symm {

// A shared library opened on first use; the driver entry points are looked up from it, so the
// host library does not link against the vendor driver
class LazyLibrary {
public:
    explicit LazyLibrary(const char* name) : name_(name) {}

    void* symbol(const char* symbol_name) {
        std::call_once(once_, [this] {
            handle_ = dlopen(name_, RTLD_LAZY | RTLD_LOCAL);
            if (handle_ == nullptr)
                panic("cannot open ", name_, ": ", dlerror());
        });
        void* symbol = dlsym(handle_, symbol_name);
        if (symbol == nullptr)
            panic("cannot find ", symbol_name, " in ", name_);
        return symbol;
    }

private:
    const char* name_;
    void* handle_ = nullptr;
    std::once_flag once_;
};

}  // namespace uni_symm

// `lazy_<name>(args...)` calls `name` from `library()`, resolved once
#define US_DECL_LAZY_FUNCTION(library, name)                                         \
    template <typename... Args>                                                      \
    inline auto lazy_##name(Args&&... args) {                                        \
        using function_t = decltype(&::name);                                        \
        static const auto function = reinterpret_cast<function_t>(library().symbol(#name)); \
        return function(std::forward<Args>(args)...);                                \
    }
