#!/usr/bin/env python3
"""Clone-shape checks: the GitHub recipe a stranger follows. No GPU, no serve."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO = SKILL_DIR.parent.parent.parent


def default_of(run_sh: str, var: str) -> str | None:
    m = re.search(rf'^{re.escape(var)}="\$\{{{re.escape(var)}:-(.*)\}}"', run_sh, re.M)
    return m.group(1) if m else None


def from_line(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith("FROM "):
            return line.split(None, 1)[1].strip()
    return None


def main() -> int:
    failures: list[str] = []

    def need(path: Path) -> None:
        if not path.exists():
            failures.append(f"missing {path.relative_to(REPO)}")

    need(REPO / "README.md")
    need(REPO / "LICENSE")
    need(REPO / "run.sh")
    need(REPO / "stop.sh")
    need(REPO / "bench_decode.py")
    need(REPO / "docker" / "Dockerfile.b12x-vision")
    need(REPO / "docker" / "plugin" / "pyproject.toml")
    need(REPO / "docker" / "plugin" / "dsv4_vision" / "__init__.py")
    need(REPO / "docker" / "plugin" / "dsv4_vision" / "vision.py")
    need(REPO / "docker" / "plugin" / "dsv4_vision" / "image_processor.py")
    need(REPO / "docker" / "plugin" / "dsv4_vision" / "model.py")
    need(REPO / "docker" / "patch_dspark_skip_bias_vl.py")
    need(REPO / "encoding" / "encoding_dsv4.py")
    need(REPO / "tests" / "test_image_grid.py")
    need(REPO / "tests" / "test_plugin_register.py")
    need(REPO / "tests" / "test_recipe_ops.py")

    run_sh_path = REPO / "run.sh"
    stop_sh_path = REPO / "stop.sh"
    if run_sh_path.exists() and not os.access(run_sh_path, os.X_OK):
        failures.append("run.sh is not executable")
    if stop_sh_path.exists() and not os.access(stop_sh_path, os.X_OK):
        failures.append("stop.sh is not executable")

    readme = (REPO / "README.md").read_text() if (REPO / "README.md").exists() else ""
    run_sh = run_sh_path.read_text() if run_sh_path.exists() else ""
    license_txt = (REPO / "LICENSE").read_text() if (REPO / "LICENSE").exists() else ""
    bench = (REPO / "bench_decode.py").read_text() if (REPO / "bench_decode.py").exists() else ""
    stop_sh = stop_sh_path.read_text() if stop_sh_path.exists() else ""

    if "MIT License" not in license_txt:
        failures.append("LICENSE is not MIT")
    if "Recipe scripts are MIT" not in readme:
        failures.append("README missing recipe MIT line")
    if "glm53-flash-nvfp4" in stop_sh:
        failures.append("stop.sh mentions glm53-flash-nvfp4")

    for snippet in (
        "docker build -f docker/Dockerfile.b12x-vision",
        "./run.sh",
        "./stop.sh",
        "python3 bench_decode.py",
        "http://127.0.0.1:8000/v1/chat/completions",
        "SPEC=dspark",
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "dsv4-flash-vision-sm121",
        "Say hello in one sentence.",
        "thinking",
        "eugr/spark-vllm-b12x:latest",
        "eugr/spark-vllm-b12x:latest@sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd",
        "bias_vl",
        "is not a multimodal model",
        "SNAPSHOT_SHA",
    ):
        if snippet not in readme:
            failures.append(f"README missing {snippet!r}")

    expected = {
        "IMAGE": "dsv4-flash-vision-sm121",
        "PORT": "8000",
        "MAX_MODEL_LEN": "1048576",
        "MAX_NUM_SEQS": "2",
        "KV_CACHE_MEMORY": "12884901888",
        "BLOCK_SIZE": "256",
        "SPEC": "dspark",
        "SERVED_NAME": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "CONTAINER_NAME": "dsv4-flash-vision-exp",
        "NUM_SPECULATIVE_TOKENS": "6",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "KV_CACHE_DTYPE": "fp8",
        "MASTER_PORT": "29522",
        "LOAD_FORMAT": "auto",
        "MOE_BACKEND": "b12x",
    }
    for var, want in expected.items():
        got = default_of(run_sh, var)
        if got != want:
            failures.append(f"run.sh default {var}={got!r} want {want!r}")
        if want not in readme:
            failures.append(f"README does not mention run.sh default {var}={want}")

    if "Do not use stock vllm/vllm-openai on sm_121" not in run_sh:
        failures.append("run.sh no longer refuses the stock image")
    if "FORCE_UNSAFE_CTX" not in run_sh or "cannot hold --max-model-len" not in run_sh:
        failures.append("run.sh no longer refuses a 1M window on the fp8 pin")
    if '"method":"dspark"' not in run_sh:
        failures.append("run.sh missing dspark speculative method")
    if "glm53-flash-nvfp4" not in run_sh:
        failures.append("run.sh lost the glm53-flash-nvfp4 occupancy guard")
    if "CONTAINER_NAME:-dsv4-flash-vision-exp" not in run_sh:
        failures.append("run.sh default container is not dsv4-flash-vision-exp")

    dockerfile = REPO / "docker/Dockerfile.b12x-vision"
    if dockerfile.exists():
        got = from_line(dockerfile)
        want_from = (
            "eugr/spark-vllm-b12x:latest@"
            "sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd"
        )
        if got != want_from:
            failures.append(f"Dockerfile.b12x-vision FROM {got!r} want {want_from!r}")
        df_text = dockerfile.read_text()
        if 'ENTRYPOINT ["vllm", "serve"]' not in df_text:
            failures.append("Dockerfile.b12x-vision missing ENTRYPOINT [\"vllm\", \"serve\"]")
        if "patch_dspark_skip_bias_vl.py" not in df_text:
            failures.append("Dockerfile.b12x-vision missing patch_dspark_skip_bias_vl.py")

    if '"prose"' not in bench or '"structured"' not in bench:
        failures.append("bench_decode.py missing prose/structured PHASES")
    if "--phase" not in bench or "chat/completions" not in bench:
        failures.append("bench_decode.py missing --phase or completions URL")

    py = sys.executable
    grid = subprocess.run(
        [py, str(REPO / "tests" / "test_image_grid.py")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if grid.returncode != 0:
        failures.append(f"test_image_grid.py failed: {grid.stderr.strip() or grid.stdout.strip()}")
    plug = subprocess.run(
        [py, str(REPO / "tests" / "test_plugin_register.py")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if plug.returncode != 0:
        failures.append(f"test_plugin_register.py failed: {plug.stderr.strip() or plug.stdout.strip()}")
    enc_a = REPO / "encoding" / "encoding_dsv4.py"
    enc_b = REPO / "docker" / "plugin" / "dsv4_vision" / "encoding_dsv4.py"
    if enc_a.exists() and enc_b.exists() and enc_a.read_bytes() != enc_b.read_bytes():
        failures.append("encoding/encoding_dsv4.py differs from docker/plugin/dsv4_vision/encoding_dsv4.py")
    ops = subprocess.run(
        [py, str(REPO / "tests" / "test_recipe_ops.py")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if ops.returncode != 0:
        failures.append(f"test_recipe_ops.py failed: {ops.stderr.strip() or ops.stdout.strip()}")

    val = subprocess.run(
        ["bash", str(run_sh_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATE_ONLY": "1"},
    )
    if val.returncode != 0 or "validate-only" not in val.stdout:
        failures.append(
            f"VALIDATE_ONLY=1 ./run.sh failed: rc={val.returncode} {val.stderr.strip() or val.stdout.strip()}"
        )
    refuse = subprocess.run(
        ["bash", str(run_sh_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATE_ONLY": "1", "MAX_MODEL_LEN": "2097152"},
    )
    if refuse.returncode == 0 or "cannot hold --max-model-len" not in refuse.stderr:
        failures.append("VALIDATE_ONLY=1 MAX_MODEL_LEN=2097152 did not refuse a window above 1M")

    print(f"repo={REPO}")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"result=fail n={len(failures)}")
        return 1
    print("result=pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
