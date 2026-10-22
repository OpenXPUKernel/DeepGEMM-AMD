#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace uni_symm {

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message) : std::runtime_error("uni_symm: " + message) {}
};

template <typename... Args>
[[noreturn]] inline void panic(const Args&... args) {
    std::ostringstream stream;
    (stream << ... << args);
    throw Exception(stream.str());
}

}  // namespace uni_symm

#define US_ASSERT(cond, ...)                                                                   \
    do {                                                                                       \
        if (!(cond))                                                                           \
            ::uni_symm::panic("assertion `" #cond "` failed (", __FILE__, ":", __LINE__, ") ", \
                               ##__VA_ARGS__);                                                 \
    } while (0)
