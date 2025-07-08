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

ARG SGLANG_VERSION="latest"

FROM lmsysorg/sglang:${SGLANG_VERSION}
WORKDIR /workspace

# GCSfuse components (used to provide shared storage, not intended for high performance)
RUN apt update && apt install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    cmake \
    dnsutils \
  && echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get install --yes google-cloud-cli \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && mkdir /gcs

RUN apt update && apt install --yes ubuntu-drivers-common

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]