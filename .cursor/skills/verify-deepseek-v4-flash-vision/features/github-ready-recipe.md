# GitHub-ready recipe

A stranger cloning this repo can follow `README.md` to pull the B12X base image, build `dsv4-flash-vision-sm121`, start the 2× Spark serve, smoke the API, bench decode, and stop.

## Sub-features

- `recipe-files` ships `README.md`, `LICENSE`, `run.sh`, `stop.sh`, `bench_decode.py`, `docker/Dockerfile.b12x-vision`, and the vision plugin.
- `recipe-exec` keeps `run.sh` and `stop.sh` executable.
- `recipe-defaults` documents the same `IMAGE`, port, context, KV pin, spec, and served name that `run.sh` defaults to.
- `recipe-images` documents `docker build -f docker/Dockerfile.b12x-vision`.
- `recipe-license` states MIT for the scripts and the model.

## How to get to it (user POV)

- Clone the GitHub repo and open `README.md`.
- Run the documented `docker build -f docker/Dockerfile.b12x-vision` command.
- Run `./run.sh`, the smoke `curl`, `python3 bench_decode.py`, and `./stop.sh` as written in `README.md`.

## Driving it with verify-deepseek-v4-flash-vision

Preconditions:

- Working directory is the repo root.
- The serve does not need to be up.

- **Lint the clone.** Run `python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/recipe-lint.py`. Exit code `0` and stdout contain `result=pass`.

## Gotchas

- Local Docker images being absent is not a recipe failure. The README tells the user to build them; lint does not require `docker image inspect`.
- `glm53-flash-nvfp4` must keep running if it is already up. This recipe uses a different container name and refuses to start beside it.
- Do not serve or bench while GLM occupies the GPUs.
