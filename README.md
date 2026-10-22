# tkt-sglang

The SGLang inference server behind the LLM Gateway. It is the `sglang`
optional component of Thinkube.

[SGLang](https://github.com/sgl-project/sglang) serves Hugging Face
checkpoints on NVIDIA GPUs, the same models as vLLM. It answers the OpenAI
API and Anthropic's Messages API.

## What it does

The container runs `python3 -m sglang.launch_server` on port 8000 with the
model the LLM gateway names.

- **The model comes from MLflow.** The entrypoint finds the latest MLflow
  version of `MODEL_ID` and serves its checkpoint under `/mlflow-models`.
- **The gateway's plan sets the server.** The load's context is
  `--context-length`; the memory share the gateway planned
  (`VLLM_GPU_MEMORY_UTILIZATION`) is `--mem-fraction-static`; a model split
  over several GPUs runs with `--tp-size`.
- **Reasoning and tools follow the catalog.** A model's `reasoning_format`
  is SGLang's `--reasoning-parser`. A model with `tool_use` gets
  `--tool-call-parser`: `qwen3_coder` for the Qwen 3.x family, or the
  parser the catalog's `tool_call_parser` names. A model with tools and
  neither stops with an error that says what to set.
- **Speculative decoding is not taken from the catalog.** The catalog's
  `speculative_config` is vLLM's; a load that carries one stops with an
  error. SGLang's own flags can be given in `server_args`.
- **The catalog can add arguments.** A model's `server_args` are added to
  the command.

## How it reaches a user

It is installed and removed from the Optional Components page in
thinkube-control, with no pods until a model is loaded. `spec.llm_backend`
in `thinkube.yaml` declares it a `performance` backend of the LLM Gateway of
type `sglang` that answers Anthropic requests itself. A model runs on it when
its catalog entry lists `sglang` in `server_type`.

## Image

The container is built on SGLang's `lmsysorg/sglang:v0.5.21-cu130-runtime`
image (CUDA 13.0, amd64 and arm64), mirrored into Harbor as
`library/sglang` from `thinkube-metadata/mirror_images.json`. It adds curl
and jq for the entrypoint.

## Licence

This template is MIT licensed. SGLang is Apache 2.0 licensed. Each model has
its own licence, stated in its model card.
