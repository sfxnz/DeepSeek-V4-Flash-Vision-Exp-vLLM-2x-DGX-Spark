# DeepSeek-V4-Flash-Vision-Exp · vLLM · 2× DGX Spark

Serve [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

284B total / 13B active MoE plus a native 32-layer ViT (~0.47B BF16). Experts are MXFP4. Attention is FP8. Native context is 1,048,576. This recipe serves `--max-model-len` 1048576 with fused DSpark (6 speculative tokens) on the B12X GB10 image.

Stock `vllm/vllm-openai` does not run MXFP4 CSA/HCA on sm_121. Build `dsv4-flash-vision-sm121` from `eugr/spark-vllm-b12x:latest` first. The Dockerfile pins `eugr/spark-vllm-b12x:latest@sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd`. The overlay `ENTRYPOINT` is `vllm serve`. The B12X base image has no CMD, so a snapshot path as argv0 execs a directory.

Pinned snapshot: `86f746b36186f0e567729a5c06a8c918caba82a9`.

## Measured on 2× DGX Spark (L.A.I.L lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens, 3-run median. `max-num-seqs=2`, fp8 KV pinned at 12 GiB (`12884901888`), context 1048576, DSpark-6, CUDA graphs. Prose is the low-acceptance regime. Structured (count 1→200) is the high-acceptance regime.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 27.0 | 27.0 | 0.37 s |
| prose | 2 | 20.6 | 39.8 | 0.37 s |
| structured | 1 | 86.9 | 86.9 | 0.34 s |
| structured | 2 | 66.0 | 130.8 | 0.51 s |
<!-- END generated measured -->

Engine log: GPU KV cache size 1,250,741 tokens, 1.19× concurrency at 1,048,576. vLLM needs 11.04 GiB for one 1M request. An 8 GiB pin estimates max len 289024. Do not use a 26 GiB community pin on this UMA leftover.

DSpark-5 is rejected (`num_speculative_tokens` must be divisible by `n_predict=3`). DSpark-6 boots. Draft acceptance on this bench: prose ~0.15, structured ~0.89.

`VLLM_USE_B12X_MHC` stays off. Vision-Exp `rms_norm_eps` is `1e-20`. The B12X fused Gram mHC kernel only accepts `1e-6`.

Prefill from unique-salt needles, thinking off: 13,349 prompt tokens at 2134 tok/s (hit), 53,349 at 1924 tok/s (hit), 200,013 at 1780 tok/s (miss, model repeated the filler). A 1M needle was not run. Host available RAM after boot is about 8 GiB. Do not raise `MAX_NUM_SEQS` on this pin.

`python3 bench_decode.py` repeats both phases at c=1,2.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes
- About 160 GiB free disk per node for the weights
- SSH from the head node to the worker (`spark2` in this lab)
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up.

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Build the image

On both nodes, from this repo:

```bash
docker pull eugr/spark-vllm-b12x:latest@sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd
docker build -f docker/Dockerfile.b12x-vision -t dsv4-flash-vision-sm121 docker
```

The Dockerfile starts from `eugr/spark-vllm-b12x:latest@sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd` and installs the official ViT/aligner as a vLLM plugin (`VLLM_PLUGINS=dsv4_vision`). The wrapper subclasses `DeepseekV4ForCausalLM` and `SupportsMultiModal` so DSpark still sees `lm_head` and `get_mtp_target_hidden_states`. A compose-style wrapper that hides those attributes collapses draft acceptance. `run.sh` refuses a missing image. Do not use stock vllm/vllm-openai on sm_121.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
./run.sh
```

The head script copies itself to `spark2`, starts the worker, waits 25s, then starts rank 0. First boot is weight load plus warmup. `run.sh` forwards `SNAPSHOT_SHA`, `HF_CACHE`, `MODEL`, and `VLLM_USE_B12X_MHC` to the worker. Download uses `--revision "$SNAPSHOT_SHA"`. If `ORCHESTRATE=auto`, `ROLE=head`, `NNODES>1`, and SSH to `WORKER_HOST` fails, the script exits 1. It does not start a TP=2 head rank alone.

If SSH is not set up, start the worker yourself, then the head:

```bash
# spark2
ROLE=worker ./run.sh

# spark1
ROLE=head ./run.sh
```

Text smoke:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"thinking": false, "reasoning_effort": "low"}
  }'
```

Vision is wired. `image_url` is accepted. The plugin registers `SupportsMultiModal` and `MULTIMODAL_REGISTRY`. A POST must not return HTTP 400 `is not a multimodal model`. After `./run.sh`, run `python3 smoke_vision.py`. Images use OpenAI `image_url` blocks. The placeholder is `<｜deepseek_image｜>`. Cap is 384 tokens per image.

Every MoE layer also ships `ffn.gate.bias_vl` (43 backbone + 3 MTP). vLLM's DeepseekV4 mapper only maps `ffn.gate.bias` to `e_score_correction_bias`. Image-token expert routing is not on the B12X text path until a later hillclimb patches the router. Text decode is unaffected.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this image? One sentence."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BASE64"}}
    ]}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"thinking": false, "reasoning_effort": "low"}
  }'
```

Stop both ranks from the head:

```bash
./stop.sh
```

`ORCHESTRATE=auto` (default) also stops the worker over SSH. If that probe fails, `stop.sh` prints to stderr and exits 1. `ORCHESTRATE=0` stops only the local container. The worker hostname is `WORKER_HOST`, else `$PWD/.run-state/worker_host`, else `spark2`.

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `dsv4-flash-vision-sm121` |
| Model | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 1048576 |
| `--max-num-seqs` | 2 |
| `--max-num-batched-tokens` | 8192 |
| `--kv-cache-dtype` | `fp8` |
| `--kv-cache-memory` | `12884901888` |
| `--moe-backend` | `b12x` |
| `--block-size` | 256 |
| CUDA graphs | on, `FULL_AND_PIECEWISE` (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Speculative | DSpark-6 (`SPEC=dspark`) |
| Load format | `auto` (safetensors). `LOAD_FORMAT=instanttensor` is opt-in |
| Tokenizers / tools / reasoning | `deepseek_v4` |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| API | `http://<head>:8000/v1` |
| Container | `dsv4-flash-vision-exp` |
| Master port | 29522 |
<!-- END generated defaults -->

`--kv-cache-memory 12884901888` is the measured 12 GiB pin (1.19× at 1M). `run.sh` refuses `--max-model-len` above 1048576 on fp8, and refuses a pin below 12 GiB when the window is above 289024, unless `FORCE_UNSAFE_CTX=1`. Official Spark occupancy is `MAX_NUM_SEQS=8`. Default here stays two sequences. Host headroom after boot is about 8 GiB.

There is no Jinja chat template. Use `--tokenizer-mode deepseek_v4` and `chat_template_kwargs.thinking`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=1048576
export MAX_NUM_SEQS=2
```

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Some Spark cookbooks use `enp1s0f0np0` / `rocep1s0f0`. This lab uses `enp1s0f1np1` / `rocep1s0f1`.

## Repeat the decode bench

```bash
python3 bench_decode.py                    # both phases, c=1,2, 3 runs
python3 bench_decode.py --phase structured # one phase only
```

## Logs

```bash
docker logs -f dsv4-flash-vision-exp
ssh spark2 docker logs -f dsv4-flash-vision-exp
```

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (MIT).
