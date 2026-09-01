#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_FROM = (
    "eugr/spark-vllm-b12x:latest@"
    "sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd"
)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FORCE_UNSAFE_CTX", None)
    env.update(extra)
    return env


def _run_sh(**extra: str) -> subprocess.CompletedProcess[str]:
    env = _env(**extra)
    env["VALIDATE_ONLY"] = "1"
    return subprocess.run(
        [str(ROOT / "run.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )


def _func_body(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{(.*?)^\}}", src, re.M | re.S)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(1)


class RecipeOpsTests(unittest.TestCase):
    def test_stop_sh_ssh_probe_has_else_exit_1(self) -> None:
        stop = _read("stop.sh")
        self.assertRegex(
            stop,
            r"ssh -o BatchMode=yes.*\n(?:.*\n)*?\s+else\n(?:.*\n)*?\s+exit 1",
        )
        self.assertIn(">&2", stop)

    def test_stop_sh_reads_worker_host_state_then_default(self) -> None:
        stop = _read("stop.sh")
        self.assertIn(".run-state/worker_host", stop)
        self.assertLess(stop.find(".run-state/worker_host"), stop.find('WORKER_HOST:-spark2'))

    def test_stop_sh_orchestrate_zero_is_local_only(self) -> None:
        stop = _read("stop.sh")
        self.assertRegex(stop, re.compile(r'ORCHESTRATE" == "0".*exit 0', re.S))
        proc = subprocess.run(
            [str(ROOT / "stop.sh")],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(
                ORCHESTRATE="0",
                CONTAINER_NAME="dsv4-ops-test-no-such-container",
            ),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_stop_sh_ssh_probe_fails_loud_when_not_spark2(self) -> None:
        host = socket.gethostname().split(".", 1)[0].lower()
        if host.startswith("spark2"):
            self.skipTest("this host is spark2")
        proc = subprocess.run(
            [str(ROOT / "stop.sh")],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(
                ORCHESTRATE="auto",
                WORKER_HOST="no-such-host-xyz",
                CONTAINER_NAME="dsv4-ops-test-no-such-container",
            ),
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_run_sh_worker_ssh_forwards_snapshot_and_revision(self) -> None:
        run = _read("run.sh")
        ssh_idx = run.find('ssh "$WORKER_HOST"')
        self.assertGreater(ssh_idx, 0)
        ssh_block = run[ssh_idx : ssh_idx + 2500]
        self.assertIn("SNAPSHOT_SHA='$SNAPSHOT_SHA'", ssh_block)
        self.assertIn("HF_CACHE='$HF_CACHE'", ssh_block)
        self.assertIn("MODEL='$MODEL'", ssh_block)
        self.assertIn("VLLM_USE_B12X_MHC=", ssh_block)
        self.assertIn('--revision "$SNAPSHOT_SHA"', run)
        self.assertIn(".run-state/worker_host", run)
        self.assertNotIn("starting local rank only", run)

    def test_resolve_model_does_not_fall_back_to_hub_id(self) -> None:
        body = _func_body(_read("run.sh"), "resolve_model")
        self.assertNotIn("$MODEL", body)
        self.assertIn("$SNAPSHOT_IN_CONTAINER", body)

    def test_dockerfile_from_is_digest_pinned(self) -> None:
        from_line = ""
        for line in _read("docker/Dockerfile.b12x-vision").splitlines():
            if line.startswith("FROM "):
                from_line = line.split(None, 1)[1].strip()
                break
        self.assertEqual(from_line, PINNED_FROM)
        self.assertIn("@sha256:", from_line)

    def test_head_preflight_before_worker_scp(self) -> None:
        run = _read("run.sh")
        idx = run.find('ORCHESTRATE" == "auto" && "$ROLE" == "head"')
        self.assertGreater(idx, 0)
        block = run[idx:]
        scp = block.find("scp ")
        self.assertGreater(scp, 0)
        self.assertGreater(block.find("refuse_foreign_serve"), -1)
        self.assertGreater(block.find("refuse_busy_port"), -1)
        self.assertLess(block.find("refuse_foreign_serve"), scp)
        self.assertLess(block.find("refuse_busy_port"), scp)
        wait = _func_body(run, "wait_ready")
        self.assertIn("$SERVED_NAME", wait)

    def test_validate_only_defaults_pass(self) -> None:
        proc = _run_sh()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validate-only", proc.stdout)

    def test_validate_only_refuses_max_num_seqs(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="8")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_spec_tokens_not_divisible_by_3(self) -> None:
        proc = _run_sh(NUM_SPECULATIVE_TOKENS="5")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_26gib_kv_pin(self) -> None:
        proc = _run_sh(KV_CACHE_MEMORY="27917287424")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_window_above_1m(self) -> None:
        proc = _run_sh(MAX_MODEL_LEN="2097152")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("cannot hold --max-model-len", proc.stderr)

    def test_gitignore_run_state(self) -> None:
        self.assertIn(".run-state/", _read(".gitignore"))

    def test_smoke_sh_curl_sf_requires_content(self) -> None:
        smoke = _read(".cursor/skills/verify-deepseek-v4-flash-vision/scripts/smoke.sh")
        self.assertIn("curl -sf", smoke)
        self.assertNotIn("smoke_vision.py", smoke)
        self.assertNotIn("smoke_tools.py", smoke)
        self.assertIn("choices", smoke)
        self.assertIn("content", smoke)

    def test_bench_wave_fails_if_any_stream_fails(self) -> None:
        bench = self._load_bench()
        n = {"i": 0}

        def flaky(*_a, **_k):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("nope")
            return {
                "ttft_s": 0.1,
                "total_s": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 10,
                "decode_tok_s": 5.0,
                "chunks": 2,
            }

        bench.stream_one = flaky
        with self.assertRaises(RuntimeError):
            bench.wave("http://127.0.0.1:9/v1/chat/completions", "m", "p", 8, 2)

    def test_bench_wave_fails_on_zero_completion_tokens(self) -> None:
        bench = self._load_bench()

        def zero(*_a, **_k):
            return {
                "ttft_s": 0.1,
                "total_s": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "decode_tok_s": 0.0,
                "chunks": 0,
            }

        bench.stream_one = zero
        with self.assertRaises(RuntimeError):
            bench.wave("http://127.0.0.1:9/v1/chat/completions", "m", "p", 8, 1)

    def _load_bench(self):
        path = ROOT / "bench_decode.py"
        spec = importlib.util.spec_from_file_location("bench_decode_ops", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


if __name__ == "__main__":
    unittest.main()
