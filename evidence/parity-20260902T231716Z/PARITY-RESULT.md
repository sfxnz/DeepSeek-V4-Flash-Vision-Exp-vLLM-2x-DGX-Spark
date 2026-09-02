# Kit parity run — DeepSeek-V4-Flash-Vision-Exp — 2026-09-02T23:17Z

Result: **bench PASS, probes FAIL on one pre-existing gate** (`needle-120000`). Serve booted on the rebuilt image, shipped defaults, no env overrides, lease `pair` unit `parity-deepseek`. Reference: `evidence/rebench-20260902T202935Z/bench.json`.

Image: rebuilt this run from `docker/` on `main` (`evidence/rebuild-20260902T224800Z/`, `plugin_check=PASS`). `image-check.txt`: rank0 ran `sha256:92edf74f…` (spark1), rank1 ran `sha256:a819a1e4…` (spark2), both the new ids. `snapshot-check.txt` and `run.log`: pinned snapshot `86f746b36186f0e567729a5c06a8c918caba82a9` on both nodes, no download.

Boot: `./run.sh` 23:17:27Z → `Ready` 23:26:17Z (8m50s). `doctor.txt`: `status=ready worker=up`. `oomkilled.txt`: both ranks `OOMKilled=false`. `stop.txt`: both ranks stopped; `cleanup.txt`: both nodes clean, lease released.

Probes (`probes.json`, `"failed": 1`): smoke PASS, count PASS (171 consecutive), thinking_off PASS, tool_call PASS (`get_weather`), hermes_two_turn PASS, vision PASS (reply `Red`), needle: 8000 PASS (13,349 prompt tokens, hit), 32000 PASS (53,349, hit), 120000 **FAIL** `needle_miss` (200,013 prompt tokens, 1744 tok/s prefill; the model repeated the filler). The README on `main` already records this exact miss ("200,013 at 1780 tok/s (miss, model repeated the filler)"); `recipe.yaml probes: needle: [8000, 32000, 120000]` enables it as a pass gate anyway, so the probe set cannot pass on `main` behaviour. Not a regression from the rebuild or the kit.

Bench (`bench_compare.py`, ±5% on decode and aggregate), one run, all rows PASS:

| Row | ref decode | parity | ref aggregate | parity | verdict |
|---|---:|---:|---:|---:|---|
| prose c=1 | 26.2 | 25.4 (−3.1%) | 26.2 | 25.4 (−3.0%) | PASS |
| prose c=2 | 20.5 | 20.6 (+0.6%) | 40.5 | 40.8 (+0.9%) | PASS |
| structured c=1 | 88.2 | 86.7 (−1.6%) | 88.2 | 86.7 (−1.6%) | PASS |
| structured c=2 | 68.6 | 66.0 (−3.8%) | 131.6 | 127.0 (−3.5%) | PASS |

`recipe.yaml measured:` is unchanged.
