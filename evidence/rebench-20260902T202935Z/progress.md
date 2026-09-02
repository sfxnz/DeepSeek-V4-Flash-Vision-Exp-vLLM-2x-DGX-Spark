# DeepSeek-V4-Flash-Vision-Exp re-bench 20260902T202935Z

Head spark1, worker spark2. Serve found already running (started 2026-09-01T12:46Z, up ~32 h, no lease). User decision: bench it in place after confirming it runs the recipe defaults, then stop it.

- 2026-09-02T20:29:35Z gate 1 ok: `## main...origin/main` clean, HEAD a34220f == origin/main
- 2026-09-02T20:29:35Z gate 2 (amended): only GPU containers are dsv4-flash-vision-exp on both nodes; `conduit` (Matrix homeserver, no GPU) ignored per coordinator
- 2026-09-02T20:29:35Z gate 3 ok: no leases; gate 4 ok: image dsv4-flash-vision-sm121 and snapshot 86f746b present on both nodes
- 2026-09-02T20:29:35Z lease acquired: pair, ttl 3h, unit rebench-deepseek
- 2026-09-02T20:33:10Z serve-config-check.md: all run.sh flags/env/tag/snapshot/served-name match defaults on both ranks. Caveat: local images predate PRs #1-#3; spark2 plugin is pre-#2 (see image-drift.diff). Proceeding in place per user decision.
- 2026-09-02T20:33:31Z smoke: exit=0 (.cursor/skills/verify-deepseek-v4-flash-vision/scripts/smoke.sh)
- 2026-09-02T20:33:32Z tools: exit=0 (python3 smoke_tools.py)
- 2026-09-02T20:33:33Z vision: exit=0 (python3 smoke_vision.py)
- 2026-09-02T20:33:47Z needle-8000: exit=0 (python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/needle_probe.py --prompt-tokens 8000)
- 2026-09-02T20:34:08Z bench_decode.py (defaults: both phases, c=1 2, runs=3, max_tokens=200): exit=0, finished 2026-09-02T20:34:54Z
- 2026-09-02T20:35:18Z bench.json written from bench.txt SUMMARY
- 2026-09-02T20:35:18Z needle-32000: exit=0 (python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/needle_probe.py --prompt-tokens 32000)
- 2026-09-02T20:36:00Z needle-120000: exit=1 (python3 .cursor/skills/verify-deepseek-v4-flash-vision/scripts/needle_probe.py --prompt-tokens 120000)
- 2026-09-02T20:38:12Z engine log tails saved (rank0 500 lines, rank1 156 lines), OOMKilled false both ranks
- 2026-09-02T20:38:59Z ./stop.sh ok; docker ps: spark1 only conduit, spark2 empty; lease pair released
- 2026-09-02T20:39:41Z redacted HF_TOKEN/HUGGING_FACE_HUB_TOKEN values in inspect-rank*.json before commit
- 2026-09-02T20:40:30Z paperwork: branch agent/rebench-2026-09-02, recipe.yaml rows from bench.json, README acceptance line ~0.14/~0.88 from bench.json, kit/render.py --check --strict pass, recipe-lint pass
