# Copyright 2026 Google LLC
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

# Serving image for Nemotron-3-Ultra 550B NVFP4 on A4X-MAX (GB300, arm64):
# the SGLang CUDA 13 runtime with the ai-dynamo SGLang backend and a pinned
# FlashInfer. The install steps are the same ones the measured runs used.
#
# Intended dependency overrides:
# * ai-dynamo[sglang]==1.4.2 pins sglang==0.5.16, which replaces the SGLang
#   that ships in the base image.
# * flashinfer-python is then force-installed at 0.6.18 with --no-deps,
#   overriding the flashinfer-python 0.6.14 pin of sglang 0.5.16. --no-deps
#   keeps sglang's nvidia-cutlass-dsl==4.6.0 even though FlashInfer 0.6.18
#   declares nvidia-cutlass-dsl>=4.6.2a0. `pip check` reports both
#   conflicts, plus a few inherited from the base image. The output is
#   recorded in /opt/dynamo-sglang-n3u-pip-check.txt and does not fail the
#   build.
ARG SGLANG_BASE_IMAGE="lmsysorg/sglang:v0.5.19-cu130-runtime"
ARG DYNAMO_VERSION="1.4.2"
ARG FLASHINFER_VERSION="0.6.18"

FROM ${SGLANG_BASE_IMAGE}
ARG DYNAMO_VERSION
ARG FLASHINFER_VERSION

RUN python3 -m pip install --no-cache-dir "ai-dynamo[sglang]==${DYNAMO_VERSION}" \
 && python3 -m pip install --no-cache-dir --force-reinstall --no-deps "flashinfer-python==${FLASHINFER_VERSION}" \
 && python3 -c "from importlib.metadata import version; print(*(f'{p}=={version(p)}' for p in ('ai-dynamo', 'sglang', 'flashinfer-python', 'nvidia-cutlass-dsl')))" \
 && python3 -m pip freeze > /opt/dynamo-sglang-n3u-packages.txt \
 && (python3 -m pip check > /opt/dynamo-sglang-n3u-pip-check.txt 2>&1 || true)

# The Dynamo Helm chart supplies the command; keep the image entrypoint neutral.
ENTRYPOINT []
