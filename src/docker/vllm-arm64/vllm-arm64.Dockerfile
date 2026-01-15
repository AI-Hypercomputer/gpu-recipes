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

ARG VLLM_VERSION="v0.14.0rc1"
ARG NGC_VLLM_VERSION="25.12.post1-py3"

FROM nvcr.io/nvidia/vllm:${NGC_VLLM_VERSION}

ARG VLLM_VERSION
ARG NGC_VLLM_VERSION

# Set up ssh for multi-host
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
        git openssh-server wget iproute2 vim build-essential \
        cmake gdb python3 python3.12-dbg curl \
  protobuf-compiler libprotobuf-dev rsync libssl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN cd /etc/ssh/ && sed --in-place=".bak" "s/#Port 22/Port 222/" sshd_config && \
    sed --in-place=".bak" "s/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/" sshd_config
RUN ssh-keygen -t rsa -b 4096 -q -f /root/.ssh/id_rsa -N ""
RUN touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# Install gcloud CLI (Production Fix for gsutil)
RUN apt-get update && apt-get install -y lsb-release curl gnupg
# Add Google Cloud source list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# Install the full SDK (which includes gcloud, gsutil, and gke-gcloud-auth-plugin)
RUN apt-get update -y && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean

# Set PATH to include gsutil binaries (correct path for apt install)
ENV PATH="/usr/lib/google-cloud-sdk/bin:${PATH}"

WORKDIR /workspace
COPY ray_init.sh /workspace/ray_init.sh

RUN echo "Cloning vLLM version: ${VLLM_VERSION}" && \
    git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    (git checkout "${VLLM_VERSION}" 2>/dev/null || true) && \
    mv vllm vllm_1

# RUN echo "Cloning vLLM" && \
#     git clone https://github.com/vllm-project/vllm.git && \
#     cd vllm && \
#     mv vllm vllm_1

ENTRYPOINT [ "/bin/bash" ]