#include "omni_vision_device_binding.h"
#include <cassert>
#include <iostream>

int main() {
    assert(omni_vision_require_pci("00000000:64:00.0", "0000:64:00.0") == "00000000:64:00.0");
    assert(omni_vision_require_pci("ABCD:EF:1F.7", "0000abcd:ef:1f.7") == "0000abcd:ef:1f.7");
    const char * invalid[] = {nullptr, "", "64:00.0", "0000:64:00.0-v0", "0000:64:20.0",
        "0000:64:00.8", "0000:64:00.a", "0000:gg:00.0", "0000:64:00.0\n", "0000:64:00.0 extra"};
    for (auto id : invalid) {
        bool rejected = false;
        try { omni_vision_require_pci("0000:64:00.0", id); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
        rejected = false;
        try { omni_vision_require_pci(id, "0000:64:00.0"); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { omni_vision_require_pci("0000:65:00.0", "0000:64:00.0"); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    std::cout << "PCI normalization and mismatch rejection passed\n";
}
