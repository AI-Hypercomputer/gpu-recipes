# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

ARG PYTHON_BASE_IMAGE=python:3.12-slim
FROM ${PYTHON_BASE_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir 'aiperf==0.12.0' 'transformers==4.57.3' tiktoken blobfile protobuf \
    && python3 -m pip freeze > /opt/client-packages.txt
WORKDIR /workspace
COPY scripts/ /workspace/scripts/
ENTRYPOINT []
CMD ["sleep", "infinity"]
