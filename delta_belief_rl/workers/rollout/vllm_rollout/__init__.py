# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from importlib.metadata import PackageNotFoundError, version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def version_tuple(v):
    """Convert version string to tuple for comparison.

    Handles version strings like '0.8.5', '0.8.5.post1', '0.8.5rc1', etc.
    Strips any non-numeric suffixes after the version numbers.
    """
    import re

    # Extract only the numeric version parts (e.g., "0.8.5.post1" -> "0.8.5")
    match = re.match(r"(\d+)\.(\d+)\.?(\d*)", v)
    if match:
        parts = [int(match.group(1)), int(match.group(2))]
        if match.group(3):  # Add patch version if present
            parts.append(int(match.group(3)))
        return tuple(parts)
    else:
        # Fallback: try splitting and converting what we can
        return tuple(int(x) for x in v.split(".") if x.isdigit())


package_name = "vllm"
package_version = get_version(package_name)

###
# package_version = get_version(package_name)
# [SUPPORT AMD:]
# Do not call any torch.cuda* API here, or ray actor creation import class will fail.
if "ROCM_PATH" in os.environ:
    import re

    package_version = version(package_name)
    package_version = re.match(r"(\d+\.\d+\.?\d*)", package_version).group(1)
else:
    package_version = get_version(package_name)
###

# Use proper version comparison instead of string comparison
# String comparison "0.11.0" <= "0.6.3" is True (wrong!)
# Tuple comparison (0, 11, 0) <= (0, 6, 3) is False (correct!)
if version_tuple(package_version) <= version_tuple("0.6.3"):
    vllm_mode = "customized"
    raise ImportError(
        "vllm version <= 0.6.3 is not supported. Please upgrade to vllm >= 0.6.4."
    )
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
