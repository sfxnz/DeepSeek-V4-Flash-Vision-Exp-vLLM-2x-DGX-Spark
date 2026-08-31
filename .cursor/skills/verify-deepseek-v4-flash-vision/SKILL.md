---
name: verify-deepseek-v4-flash-vision
description: Drive the DeepSeek-V4-Flash-Vision-Exp vLLM recipe (clone → B12X image → 2× DGX Spark OpenAI API). Shared host-network GPU serve — never start a second instance, never stop glm53-flash-nvfp4, never stop a container this run did not start.
---

# Verify DeepSeek-V4-Flash-Vision-Exp recipe

Repo root: parent of `.cursor/`. Defaults live in `run.sh`. Container name is `dsv4-flash-vision-exp`. Do not `docker rm` `glm53-flash-nvfp4`.

Two GPU serves cannot coexist on these Sparks. If `glm53-flash-nvfp4` is up, attach to GLM or refuse. Never `./run.sh` over a running DeepSeek container (it `docker rm -f`s only `dsv4-flash-vision-exp`). Never `./stop.sh` unless this run created `.cursor/skills/verify-deepseek-v4-flash-vision/.run-state/started`.

This phase does not serve and does not bench. Drive `python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/recipe-lint.py` until `result=pass`.
