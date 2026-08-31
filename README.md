# DeepSeek-V4-Flash-Vision-Exp · vLLM · 2× DGX Spark

Serve [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

284B total / 13B active MoE plus a native 32-layer ViT (~0.47B BF16). Experts are MXFP4. Attention is FP8. Native context is 1,048,576. This recipe serves `--max-model-len` 327680 with fused DSpark (5 speculative tokens) on the B12X GB10 image.

Stock `vllm/vllm-openai` does not run MXFP4 CSA/HCA on sm_121. Build `dsv4-flash-vision-sm121` from `eugr/spark-vllm-b12x:latest` first.

Decode bench is later. This recipe is not yet timed on this lab.

Pinned snapshot: `86f746b36186f0e567729a5c06a8c918caba82a9`.

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
docker pull eugr/spark-vllm-b12x:latest
docker build -f docker/Dockerfile.b12x-vision -t dsv4-flash-vision-sm121 docker
```

The Dockerfile starts from `eugr/spark-vllm-b12x:latest` and installs the official ViT/aligner as a vLLM plugin (`VLLM_PLUGINS=dsv4_vision`). `run.sh` refuses a missing image. Do not use stock vllm/vllm-openai on sm_121.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
./run.sh
```

The head script copies itself to `spark2`, starts the worker, waits 25s, then starts rank 0. First boot is weight load plus warmup.

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

Vision smoke. Images use OpenAI `image_url` blocks. The placeholder is `<｜deepseek_image｜>`. Cap is 384 tokens per image.

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

## Defaults

| Setting | Value |
|---|---|
| Image | `dsv4-flash-vision-sm121` |
| Model | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 327680 |
| `--max-num-seqs` | 2 |
| `--kv-cache-dtype` | `fp8` |
| `--kv-cache-memory` | `4445787956` |
| `--moe-backend` | `b12x` |
| `--block-size` | 256 |
| CUDA graphs | on, `FULL_AND_PIECEWISE` (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Speculative | DSpark-5 (`SPEC=dspark`) |
| Load format | `auto` (safetensors). `LOAD_FORMAT=instanttensor` is opt-in |
| Tokenizers / tools / reasoning | `deepseek_v4` |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| API | `http://<head>:8000/v1` |
| Container | `dsv4-flash-vision-exp` |
| Master port | 29522 |

`--kv-cache-memory 4445787956` is the lab UMA pin (4.14 GiB). Do not advertise a 1,048,576 window on this pin. `run.sh` refuses `--max-model-len` above 327680 on fp8 unless `FORCE_UNSAFE_CTX=1`. Official Spark occupancy is `MAX_NUM_SEQS=8` with `--max-num-batched-tokens 8192`. Default here is two sequences until that occupancy is measured on Vision-Exp.

There is no Jinja chat template. Use `--tokenizer-mode deepseek_v4` and `chat_template_kwargs.thinking`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=327680
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
