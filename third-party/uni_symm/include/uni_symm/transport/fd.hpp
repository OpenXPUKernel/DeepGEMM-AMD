#pragma once

#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include <uni_symm/utils/exception.hpp>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

namespace uni_symm {

// A duplicate, in this process, of descriptor `fd` of process `pid` on the same node (Linux 5.6+;
// needs ptrace access to the peer: the same user, and Yama ptrace_scope 0 or a ptracer grant)
inline int duplicate_peer_fd(const int pid, const int fd) {
    const int pidfd = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
    if (pidfd < 0)
        panic("pidfd_open(", pid, ") failed: ", std::strerror(errno));
    const int local_fd = static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, fd, 0));
    const int error = errno;
    close(pidfd);
    if (local_fd < 0)
        panic("pidfd_getfd(", pid, ", ", fd, ") failed: ", std::strerror(error),
              error == EPERM ? " (the peer process must be ptrace-accessible: same user and "
                               "/proc/sys/kernel/yama/ptrace_scope 0)" : "");
    return local_fd;
}

}  // namespace uni_symm
