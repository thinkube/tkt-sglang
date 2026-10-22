# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: MIT

# SGLang 0.5.21's serving runtime with CUDA 13.0, for amd64 and arm64,
# mirrored into Harbor (thinkube-metadata/mirror_images.json).
ARG CONTAINER_REGISTRY
FROM ${CONTAINER_REGISTRY}/library/sglang:v0.5.21-cu130-runtime

# The entrypoint finds the model in MLflow with curl and jq.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl jq \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/serve-model
RUN chmod 0755 /usr/local/bin/serve-model

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve-model"]
