# Copyright 2025 Google LLC
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

ARG VLLM_VERSION="latest"

FROM docker.io/vllm/vllm-openai:${VLLM_VERSION}
ARG VLLM_VERSION

WORKDIR /workspace
COPY ray_init.sh /workspace/ray_init.sh

RUN apt-get update && apt-get install -y --no-install-recommends pciutils

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

RUN echo "Cloning vLLM version: ${VLLM_VERSION}" && \
    git clone -b ${VLLM_VERSION} https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    mv vllm vllm_1

ENTRYPOINT [ "/bin/bash" ]