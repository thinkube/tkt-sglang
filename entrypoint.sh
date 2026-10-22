#!/bin/sh

# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: MIT

# Serves the Hugging Face checkpoint of $MODEL_ID with SGLang's OpenAI and
# Anthropic APIs on port 8000. The model was mirrored from Hugging Face into
# MLflow; its files are under /mlflow-models, and SGLang reads them once
# while it loads.

set -eu

MLFLOW_URL="http://mlflow.mlflow.svc.cluster.local:5000"

require() {
    eval "value=\${$1:-}"
    if [ -z "$value" ]; then
        echo "$1 is not set: $2" >&2
        exit 1
    fi
}

require MODEL_ID "the LLM gateway sets it when it creates the pod"
require MAX_CONTEXT_LENGTH "the LLM gateway sets it from the load's context"
require VLLM_GPU_MEMORY_UTILIZATION "the LLM gateway sets it from the model's memory plan"
for name in MLFLOW_KEYCLOAK_TOKEN_URL MLFLOW_KEYCLOAK_CLIENT_ID MLFLOW_CLIENT_SECRET \
            MLFLOW_AUTH_USERNAME MLFLOW_AUTH_PASSWORD; do
    require "$name" "the component's MLflow credentials Secret provides it"
done

# The catalog's speculative_config is vLLM's --speculative-config, which
# SGLang does not take; its speculative decoding has flags of its own.
if [ -n "${SPECULATIVE_CONFIG:-}" ]; then
    echo "SPECULATIVE_CONFIG is a vLLM setting that SGLang does not take: $SPECULATIVE_CONFIG." >&2
    echo "Load the model without speculative decoding, or give SGLang's flags in the catalog's server_args." >&2
    exit 1
fi

echo "Finding $MODEL_ID in MLflow"
token=$(curl -sf -X POST "$MLFLOW_KEYCLOAK_TOKEN_URL" \
    --data-urlencode grant_type=password \
    --data-urlencode "client_id=$MLFLOW_KEYCLOAK_CLIENT_ID" \
    --data-urlencode "client_secret=$MLFLOW_CLIENT_SECRET" \
    --data-urlencode "username=$MLFLOW_AUTH_USERNAME" \
    --data-urlencode "password=$MLFLOW_AUTH_PASSWORD" \
    --data-urlencode scope=openid | jq -r '.access_token // empty') || true
if [ -z "$token" ]; then
    echo "Keycloak at $MLFLOW_KEYCLOAK_TOKEN_URL gave no MLflow token for $MLFLOW_AUTH_USERNAME" >&2
    exit 1
fi

registered=$(echo "$MODEL_ID" | tr / -)
run_id=$(curl -sf -G "$MLFLOW_URL/api/2.0/mlflow/model-versions/search" \
    -H "Authorization: Bearer $token" \
    --data-urlencode "filter=name='$registered'" \
    | jq -r '.model_versions // [] | max_by(.version | tonumber) | .run_id // empty')
if [ -z "$run_id" ]; then
    echo "$MODEL_ID is not registered in MLflow as $registered; mirror it first" >&2
    exit 1
fi
experiment_id=$(curl -sf -G "$MLFLOW_URL/api/2.0/mlflow/runs/get" \
    -H "Authorization: Bearer $token" \
    --data-urlencode "run_id=$run_id" | jq -r .run.info.experiment_id)

model_dir="/mlflow-models/artifacts/$experiment_id/$run_id/artifacts/model"
if [ ! -f "$model_dir/config.json" ]; then
    echo "$model_dir has no config.json; SGLang serves Hugging Face checkpoints" >&2
    exit 1
fi

set -- \
    --model-path "$model_dir" \
    --served-model-name "$MODEL_ID" \
    --context-length "$MAX_CONTEXT_LENGTH" \
    --mem-fraction-static "$VLLM_GPU_MEMORY_UTILIZATION" \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 8000

if [ -n "${REASONING_FORMAT:-}" ]; then
    set -- "$@" --reasoning-parser "$REASONING_FORMAT"
fi

# Tool calls are parsed by the parser of the model's family: the Qwen 3.x
# models write the Qwen XML tool-call format, which qwen3_coder parses. The
# catalog's tool_call_parser names another.
if [ "${TOOL_USE:-}" = "true" ]; then
    parser="${TOOL_CALL_PARSER:-}"
    if [ -z "$parser" ]; then
        case "${REASONING_FORMAT:-}" in
            qwen3|qwen) parser=qwen3_coder ;;
            *)
                echo "$MODEL_ID uses tools, and no tool-call parser is known for its family" >&2
                echo "(reasoning_format '${REASONING_FORMAT:-}'); name SGLang's parser in the catalog's tool_call_parser." >&2
                exit 1
                ;;
        esac
    fi
    set -- "$@" --tool-call-parser "$parser"
fi

if [ -n "${TENSOR_PARALLEL_SIZE:-}" ] && [ "$TENSOR_PARALLEL_SIZE" != "1" ]; then
    set -- "$@" --tp-size "$TENSOR_PARALLEL_SIZE"
fi

if [ "${ENFORCE_EAGER:-}" = "true" ]; then
    set -- "$@" --disable-cuda-graph
fi

# Arguments the model's catalog entry adds (server_args), as a JSON list.
if [ -n "${SERVER_ARGS:-}" ]; then
    if ! echo "$SERVER_ARGS" | jq -e 'type == "array" and all(type == "string")' >/dev/null; then
        echo "SERVER_ARGS is not a JSON list of strings: $SERVER_ARGS" >&2
        exit 1
    fi
fi
while IFS= read -r arg; do
    if [ -n "$arg" ]; then set -- "$@" "$arg"; fi
done <<EOF_ARGS
$(echo "${SERVER_ARGS:-[]}" | jq -r '.[]')
EOF_ARGS

echo "Serving $model_dir with a $MAX_CONTEXT_LENGTH token context"
# The model is read from the mount; nothing is downloaded.
export HF_HUB_OFFLINE=1
exec python3 -m sglang.launch_server "$@"
