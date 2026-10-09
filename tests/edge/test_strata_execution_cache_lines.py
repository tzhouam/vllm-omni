# SPDX-License-Identifier: Apache-2.0
"""Cache-line extraction regressions; no runtime, build or model proof.

Build admission requires exactly one value equal to the declared control.
The synthetic records below keep physical line endings and value characters
explicit; they are not copied from a private machine's CMake cache.
"""

import pytest

from vllm_omni.engine.backends.strata_execution import strata_exec_runtime as sut

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
@pytest.mark.parametrize(
    ("key", "cache_type", "expected"),
    [
        ("CMAKE_BUILD_TYPE", "STRING", "Release"),
        ("CMAKE_CUDA_ARCHITECTURES", "STRING", "120"),
        ("CMAKE_CUDA_RUNTIME_LIBRARY", "STRING", "Static"),
        ("STRATA_ENABLE_CUDA", "BOOL", "ON"),
        ("STRATA_PORTABLE", "BOOL", "ON"),
        ("STRATA_NATIVE_EXPERTS", "BOOL", "ON"),
        ("STRATA_MMQ_KQUANTS", "BOOL", "OFF"),
        ("STRATA_BUILD_TESTS", "BOOL", "OFF"),
    ],
)
def test_control_values_accept_physical_lf_crlf_and_unterminated_last_line(key, cache_type, expected, ending):
    cache = f"// Synthetic cache excerpt\r\n{key}_EXTRA:{cache_type}=different\n{key}:{cache_type}={expected}{ending}"
    assert sut.cmake_cache_values(cache, key) == [expected]


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
@pytest.mark.parametrize(
    ("key", "cache_type", "expected"),
    [
        ("CMAKE_HOME_DIRECTORY", "INTERNAL", "C:/synthetic/source"),
        ("CMAKE_MAKE_PROGRAM", "FILEPATH", "C:/Program Files/Ninja/ninja.exe"),
        ("CMAKE_CUDA_COMPILER", "FILEPATH", '"C:/Program Files/NVIDIA CUDA/bin/nvcc.exe"'),
        ("STRATA_GGML_DIR", "PATH", "C:/synthetic/dependency=reference/llama.cpp"),
    ],
)
def test_windows_path_values_keep_drive_colons_spaces_quotes_and_equals(key, cache_type, expected, ending):
    cache = f"UNRELATED:STRING=before\r\n{key}:{cache_type}={expected}{ending}"
    assert sut.cmake_cache_values(cache, key) == [expected]


def test_value_ends_at_physical_line_and_adjacent_key_does_not_match():
    cache = (
        "CMAKE_MAKE_PROGRAM:FILEPATH=C:/synthetic/ninja.exe\r\n"
        "CMAKE_MAKE_PROGRAM_EXTRA:FILEPATH=C:/other/ninja.exe\n"
        "NEXT:STRING=not part of the path\r\n"
    )
    assert sut.cmake_cache_values(cache, "CMAKE_MAKE_PROGRAM") == ["C:/synthetic/ninja.exe"]


@pytest.mark.parametrize("second", ["Release", "Debug"])
def test_duplicate_values_cannot_satisfy_exact_singleton_admission(second):
    cache = f"CMAKE_BUILD_TYPE:STRING=Release\r\nCMAKE_BUILD_TYPE:STRING={second}\n"
    values = sut.cmake_cache_values(cache, "CMAKE_BUILD_TYPE")
    assert values == ["Release", second]
    assert values != ["Release"]


@pytest.mark.parametrize(
    "cache",
    ["", "// CMAKE_BUILD_TYPE:STRING=Release\r\n", "CMAKE_BUILD_TYPE_EXTRA:STRING=Release\n"],
)
def test_missing_exact_key_cannot_satisfy_singleton_admission(cache):
    assert sut.cmake_cache_values(cache, "CMAKE_BUILD_TYPE") == []


@pytest.mark.parametrize("value", ["Debug", "release", " Release", "Release ", "Release\t", '"Release"'])
def test_changed_or_whitespace_values_are_preserved_and_cannot_satisfy_exact_admission(value):
    values = sut.cmake_cache_values(f"CMAKE_BUILD_TYPE:STRING={value}\r\n", "CMAKE_BUILD_TYPE")
    assert values == [value]
    assert values != ["Release"]


@pytest.mark.parametrize(
    "cache",
    [
        "CMAKE_BUILD_TYPE:STRING=Release\r",
        "CMAKE_BUILD_TYPE:STRING=Re\rlease\n",
        "CMAKE_BUILD_TYPE:STRING=Release\r\r\n",
        "// unrelated bare\rcomment\nCMAKE_BUILD_TYPE:STRING=Release\r\n",
    ],
)
def test_bare_or_embedded_carriage_return_is_rejected(cache):
    with pytest.raises(sut.EvidenceError, match="^build_cache_line_endings$"):
        sut.cmake_cache_values(cache, "CMAKE_BUILD_TYPE")
