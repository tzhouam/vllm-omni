# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest


@pytest.fixture(autouse=True)
def log_test_name_before_test(request: pytest.FixtureRequest):
    print(f"--- Running test: {request.node.name}")
    yield
