#pragma once

#include <stdexcept>
#include <string>

// Normalize CUDA's four-digit domain and NVML's eight-digit domain.
inline std::string omni_vision_pci_id(const char * value) {
    if (!value) {
        throw std::runtime_error("vision PCI identity missing");
    }
    std::string id(value);
    const size_t domain = id.size() == 12 ? 4 : id.size() == 16 ? 8 : 0;
    if (!domain || id[domain] != ':' || id[domain + 3] != ':' || id[domain + 6] != '.') {
        throw std::runtime_error("vision PCI identity malformed");
    }
    for (size_t i = 0; i < id.size(); ++i) {
        if (i == domain || i == domain + 3 || i == domain + 6) {
            continue;
        }
        char & c = id[i];
        if (c >= 'A' && c <= 'F') {
            c = static_cast<char>(c - 'A' + 'a');
        }
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            throw std::runtime_error("vision PCI identity malformed");
        }
    }
    if (id[domain + 4] > '1' || id[domain + 7] > '7') {
        throw std::runtime_error("vision PCI device/function out of range");
    }
    return domain == 4 ? "0000" + id : id;
}

inline std::string omni_vision_require_pci(const char * expected, const char * observed) {
    const std::string actual = omni_vision_pci_id(observed);
    if (omni_vision_pci_id(expected) != actual) {
        throw std::runtime_error("vision selected GPU differs from admitted GPU");
    }
    return actual;
}
