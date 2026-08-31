---
name: verify-deepseek-v4-flash-vision
description: Drive the DeepSeek-V4-Flash-Vision-Exp vLLM recipe (clone → B12X image → 2× DGX Spark OpenAI API). Shared host-network GPU serve — never start a second instance, never stop glm53-flash-nvfp4, never stop a container this run did not start.
---

# Verify DeepSeek-V4-Flash-Vision-Exp recipe

Repo root: parent of `.cursor/`. Defaults live in `run.sh`. Container name is `dsv4-flash-vision-exp`. Do not `docker rm` `glm53-flash-nvfp4`.

Two GPU serves cannot coexist on these Sparks. If `glm53-flash-nvfp4` is up, attach to GLM or refuse unless the user explicitly authorized unloading GLM. Unload GLM only with that recipe's `stop.sh`. Never `docker rm` GLM from this recipe. Never `./run.sh` over a running DeepSeek container (it `docker rm -f`s only `dsv4-flash-vision-exp`). Never `./stop.sh` unless this run created `.cursor/skills/verify-deepseek-v4-flash-vision/.run-state/started`.

Drive `python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/recipe-lint.py` until `result=pass`. When GPUs are exclusive, `./run.sh`, the README smoke curl, `python3 bench_decode.py`, and `python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/needle_probe.py --prompt-tokens 8000`. Do not raise the 12 GiB KV pin or `MAX_NUM_SEQS` without watching host available RAM.
