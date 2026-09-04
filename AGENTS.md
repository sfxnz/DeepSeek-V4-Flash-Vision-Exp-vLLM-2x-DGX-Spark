# AGENTS.md — DeepSeek-V4-Flash-Vision-Exp · 2× DGX Spark

Serve `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at TP=2. Local image `dsv4-flash-vision-sm121` from the pinned B12X digest. Snapshot `86f746b`. Stock `vllm/vllm-openai` does not run MXFP4 CSA/HCA on sm_121.

Humans read [README.md](README.md).

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Default thinking is off. `chat_template_kwargs`: `thinking=false`, `reasoning_effort=low`. There is no Jinja chat template. Use `--tokenizer-mode deepseek_v4`.
- Leave `VLLM_USE_B12X_MHC` off. Vision-Exp `rms_norm_eps` is `1e-20`. The B12X fused Gram mHC kernel only accepts `1e-6`.
- Keep the plugin subclassing `DeepseekV4ForCausalLM` and `SupportsMultiModal`. A wrapper that hides `lm_head` or `get_mtp_target_hidden_states` collapses DSpark draft acceptance.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

Exits unless `FORCE_UNSAFE_CTX=1`:

- `--max-model-len` above 1048576 on fp8
- KV pin below `12884901888` (12 GiB) when the window is above 289024
- KV pin above 12 GiB
- `MAX_NUM_SEQS` above 2
- `NUM_SPECULATIVE_TOKENS` not divisible by 3 (DSpark-5 is rejected, DSpark-6 boots)

Do not raise `MAX_NUM_SEQS` on this pin. Host headroom after boot is about 8 GiB.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
```

After `./run.sh` is up:

```bash
python3 smoke_vision.py    # must not return HTTP 400 "is not a multimodal model"
python3 smoke_tools.py
python3 bench_decode.py    # both phases, c=1 and 2, 3-run median
```

Vision uses OpenAI `image_url`. Placeholder is `<｜deepseek_image｜>`. Cap is 384 tokens per image.

Do not claim a 1M needle. It was not run. 200,013 missed.

## Never touch

- Live HF tokens
- Floating `:latest` on the B12X base. Digest is pinned in `recipe.yaml` and the Dockerfile.
- Hand-edited generated README / `run.sh` blocks
