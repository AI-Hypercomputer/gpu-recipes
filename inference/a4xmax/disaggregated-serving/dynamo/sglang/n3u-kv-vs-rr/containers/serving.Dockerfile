# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Use the same CUDA 13 / ARM64 base as the measured jobs. Resolve the base
# to a digest before building an image for a published comparison.
ARG SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.19-cu130-runtime
FROM ${SGLANG_BASE_IMAGE}
RUN python3 -m pip install --no-cache-dir 'ai-dynamo[sglang]==1.4.2' \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps flashinfer-python==0.6.18 \
    && python3 -c "from importlib.metadata import version; expected={'ai-dynamo':'1.4.2','sglang':'0.5.16','flashinfer-python':'0.6.18'}; assert all(version(k)==v for k,v in expected.items()), expected" \
    && python3 -m pip freeze > /opt/serving-packages.txt
ENTRYPOINT []
