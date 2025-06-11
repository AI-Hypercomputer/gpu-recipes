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

FROM nvcr.io/nvidia/nemo:25.04
WORKDIR /app

RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
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

# Make the pods capable of speaking to each other via SSH on port 222           
RUN cd /etc/ssh/ && sed --in-place='.bak' 's/#Port 22/Port 222/' sshd_config && \
    sed --in-place='.bak' 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' sshd_config
RUN ssh-keygen -t rsa -b 4096 -q -f /root/.ssh/id_rsa -N ""                     
RUN touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys    
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
RUN pip install datasets==2.20.0 hydra-core sentencepiece transformers==4.45.1
RUN pip install "git+https://github.com/mlperf/logging.git"
RUN pip install git+https://github.com/NVIDIA/NeMo-Run.git

WORKDIR /app/training
ADD run.py docker/nemo_strategy_lib.patch /app/training/
RUN patch --directory=/opt/NeMo -p1 < nemo_strategy_lib.patch