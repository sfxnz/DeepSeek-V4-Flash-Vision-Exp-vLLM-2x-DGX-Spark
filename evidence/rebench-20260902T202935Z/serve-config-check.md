# Serve config check: running dsv4-flash-vision-exp vs recipe defaults

Generated from `inspect-rank0.json`, `inspect-rank1.json`, `image-rank*.json` in this directory, compared with `recipe.yaml` `serve.env` at HEAD a34220f and the literals in `run.sh`. Redacted: HF_TOKEN / HUGGING_FACE_HUB_TOKEN.


## rank 0 (spark1)

container started 2026-09-01T12:46:12.598569601Z, status running, OOMKilled False, image id `sha256:32624fee3e557ea91492eddf76b941b0c8949473310a27bdd6b1b8f6cbf723ae`, local `dsv4-flash-vision-sm121` id `sha256:32624fee3e557ea91492eddf76b941b0c8949473310a27bdd6b1b8f6cbf723ae`, image created 2026-09-01T13:44:51.540816941+01:00

| Setting | Expected (source) | Running | Match |
|---|---|---|---|
| image tag | `dsv4-flash-vision-sm121` (serve.env IMAGE) | `dsv4-flash-vision-sm121` | ok |
| container image id == local tag id | `same` (docker image inspect) | `same` | ok |
| argv0 snapshot path | `/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/86f746b36186f0e567729a5c06a8c918caba82a9` (serve.env SNAPSHOT_SHA) | `/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/86f746b36186f0e567729a5c06a8c918caba82a9` | ok |
| --served-model-name | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (serve.env SERVED_NAME) | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | ok |
| --tensor-parallel-size | `2` (serve.env TP) | `2` | ok |
| --nnodes | `2` (serve.env NNODES) | `2` | ok |
| --master-addr | `10.100.8.1` (serve.env HEAD_IP) | `10.100.8.1` | ok |
| --master-port | `29522` (serve.env MASTER_PORT) | `29522` | ok |
| --max-model-len | `1048576` (serve.env MAX_MODEL_LEN) | `1048576` | ok |
| --max-num-seqs | `2` (serve.env MAX_NUM_SEQS) | `2` | ok |
| --max-num-batched-tokens | `8192` (serve.env MAX_NUM_BATCHED_TOKENS) | `8192` | ok |
| --kv-cache-dtype | `fp8` (serve.env KV_CACHE_DTYPE) | `fp8` | ok |
| --kv-cache-memory | `12884901888` (serve.env KV_CACHE_MEMORY) | `12884901888` | ok |
| --gpu-memory-utilization | `0.80` (serve.env UTIL) | `0.80` | ok |
| --block-size | `256` (serve.env BLOCK_SIZE) | `256` | ok |
| --moe-backend | `b12x` (serve.env MOE_BACKEND) | `b12x` | ok |
| --linear-backend | `b12x` (run.sh literal) | `b12x` | ok |
| --attention-backend | `B12X_MLA_SPARSE` (run.sh literal) | `B12X_MLA_SPARSE` | ok |
| --speculative-config | `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}` (run.sh SPEC=dspark, NUM_SPECULATIVE_TOKENS) | `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}` | ok |
| --compilation-config (ENFORCE_EAGER=0) | `{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}` (run.sh COMPILATION_CONFIG) | `{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}` | ok |
| --max-cudagraph-capture-size | `16` (run.sh literal) | `16` | ok |
| --enforce-eager absent | `absent` (serve.env ENFORCE_EAGER=0) | `absent` | ok |
| --load-format absent | `absent` (serve.env LOAD_FORMAT=auto) | `absent` | ok |
| --tokenizer-mode | `deepseek_v4` (run.sh literal) | `deepseek_v4` | ok |
| --tool-call-parser | `deepseek_v4` (run.sh literal) | `deepseek_v4` | ok |
| --reasoning-parser | `deepseek_v4` (run.sh literal) | `deepseek_v4` | ok |
| --default-chat-template-kwargs | `{"thinking":false,"reasoning_effort":"low"}` (run.sh literal) | `{"thinking":false,"reasoning_effort":"low"}` | ok |
| --limit-mm-per-prompt | `{"image":8}` (run.sh literal) | `{"image":8}` | ok |
| --enable-prefix-caching | `present` (run.sh literal) | `present` | ok |
| --enable-auto-tool-choice | `present` (run.sh literal) | `present` | ok |
| --trust-remote-code | `present` (run.sh literal) | `present` | ok |
| env VLLM_USE_B12X_MHC | `0` (run.sh default 0) | `0` | ok |
| env VLLM_USE_BREAKABLE_CUDAGRAPH | `0` (serve.env) | `0` | ok |
| env VLLM_PLUGINS | `dsv4_vision` (run.sh literal) | `dsv4_vision` | ok |
| env NCCL_IB_HCA | `rocep1s0f1` (serve.env HCA) | `rocep1s0f1` | ok |
| env NCCL_SOCKET_IFNAME | `enp1s0f1np1` (serve.env IFACE) | `enp1s0f1np1` | ok |
| env VLLM_USE_B12X_MOE/FP8_GEMM/WO_PROJECTION/SPARSE_INDEXER/V2_MODEL_RUNNER | `1,1,1,1,1` (run.sh literal) | `1,1,1,1,1` | ok |
| env B12X_MLA_SM120_UNIFIED,B12X_MOE_FORCE_A8 | `1,1` (run.sh literal) | `1,1` | ok |
| HF cache mount | `/home/sfxnz/.cache/huggingface -> /cache/huggingface` (serve.env HF_CACHE) | `/home/sfxnz/.cache/huggingface -> /cache/huggingface` | ok |
| host network / host ipc / shm 32g | `host/host/34359738368` (run.sh literal) | `host/host/34359738368` | ok |
| rank args | `--host 0.0.0.0 --port 8000, --node-rank 0` (run.sh) | `--host 0.0.0.0 --port 8000, --node-rank 0` | ok |
| EXTRA_ARGS (flags not emitted by run.sh) | `none` (serve.env EXTRA_ARGS="") | `none` | ok |

Full argv: `["/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/86f746b36186f0e567729a5c06a8c918caba82a9", "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", "0", "--distributed-executor-backend", "mp", "--master-addr", "10.100.8.1", "--master-port", "29522", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "1048576", "--kv-cache-dtype", "fp8", "--kv-cache-memory", "12884901888", "--gpu-memory-utilization", "0.80", "--max-num-seqs", "2", "--max-num-batched-tokens", "8192", "--compilation-config", "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}", "--max-cudagraph-capture-size", "16", "--block-size", "256", "--moe-backend", "b12x", "--linear-backend", "b12x", "--attention-backend", "B12X_MLA_SPARSE", "--speculative-config", "{\"method\":\"dspark\",\"num_speculative_tokens\":6,\"draft_sample_method\":\"probabilistic\",\"attention_backend\":\"B12X_MLA_SPARSE\"}", "--tokenizer-mode", "deepseek_v4", "--tool-call-parser", "deepseek_v4", "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v4", "--reasoning-config", "{\"reasoning_parser\":\"deepseek_v4\",\"reasoning_start_str\":\"\",\"reasoning_end_str\":\"\"}", "--default-chat-template-kwargs", "{\"thinking\":false,\"reasoning_effort\":\"low\"}", "--limit-mm-per-prompt", "{\"image\":8}", "--enable-prefix-caching", "--served-model-name", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "--trust-remote-code"]`


## rank 1 (spark2)

container started 2026-09-01T12:45:47.075009813Z, status running, OOMKilled False, image id `sha256:64792dec00ded11b605211d6592ed94a87579d0c68c2a9c79ec38e518db8fe69`, local `dsv4-flash-vision-sm121` id `sha256:64792dec00ded11b605211d6592ed94a87579d0c68c2a9c79ec38e518db8fe69`, image created 2026-09-01T13:44:56.02173556+01:00

| Setting | Expected (source) | Running | Match |
|---|---|---|---|
| image tag | `dsv4-flash-vision-sm121` (serve.env IMAGE) | `dsv4-flash-vision-sm121` | ok |
| container image id == local tag id | `same` (docker image inspect) | `same` | ok |
| argv0 snapshot path | `/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/86f746b36186f0e567729a5c06a8c918caba82a9` (serve.env SNAPSHOT_SHA) | `/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/86f746b36186f0e567729a5c06a8c918caba82a9` | ok |
| --served-model-name | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (serve.env SERVED_NAME) | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | ok |
| --tensor-parallel-size | `2` (serve.env TP) | `2` | ok |
| --nnodes | `2` (serve.env NNODES) | `2` | ok |
| --master-addr | `10.100.8.1` (serve.env HEAD_IP) | `10.100.8.1` | ok |
| --master-port | `29522` (serve.env MASTER_PORT) | `29522` | ok |
| --max-model-len | `1048576` (serve.env MAX_MODEL_LEN) | `1048576` | ok |
| --max-num-seqs | `2` (serve.env MAX_NUM_SEQS) | `2` | ok |
| --max-num-batched-tokens | `8192` (serve.env MAX_NUM_BATCHED_TOKENS) | `8192` | ok |
| --kv-cache-dtype | `fp8` (serve.env KV_CACHE_DTYPE) | `fp8` | ok |
| --kv-cache-memory | `12884901888` (serve.env KV_CACHE_MEMORY) | `12884901888` | ok |
| --gpu-memory-utilization | `0.80` (serve.env UTIL) | `0.80` | ok |
| --block-size | `256` (serve.env BLOCK_SIZE) | `256` | ok |
| --moe-backend | `b12x` (serve.env MOE_BACKEND) | `b12x` | ok |
| --linear-backend | `b12x` (run.sh literal) | `b12x` | ok |
| --attention-backend | `B12X_MLA_SPARSE` (run.sh literal) | `B12X_MLA_SPARSE` | ok |
| --speculative-config | `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}` (run.sh SPEC=dspark, NUM_SPECULATIVE_TOKENS) | `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}` | ok |
| --compilation-config (ENFORCE_EAGER=0) | `{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}` (run.sh COMPILATION_CONFIG) | `{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}` | ok |
| --max-cudagraph-capture-size | `16` (run.sh literal) | `16` | ok |
| --enforce-eager absent | `absent` (serve.env ENFORCE_EAGER=0) | `absent` | ok |
| --load-format absent | `absent` (serve.env LOAD_FORMAT=auto) | `absent` | ok |
| --tokenizer-mode | `deepseek_v4` (run.sh literal) | `deepseek_v4` | ok |
| --tool-call-parser | `deepseek_v4` (run.sh literal) | `deepseek_v4` | ok |
| --reasoning-parser | `deepseek_v4` (run.sh literal) | `deepseek_v4` | ok |
| --default-chat-template-kwargs | `{"thinking":false,"reasoning_effort":"low"}` (run.sh literal) | `{"thinking":false,"reasoning_effort":"low"}` | ok |
| --limit-mm-per-prompt | `{"image":8}` (run.sh literal) | `{"image":8}` | ok |
| --enable-prefix-caching | `present` (run.sh literal) | `present` | ok |
| --enable-auto-tool-choice | `present` (run.sh literal) | `present` | ok |
| --trust-remote-code | `present` (run.sh literal) | `present` | ok |
| env VLLM_USE_B12X_MHC | `0` (run.sh default 0) | `0` | ok |
| env VLLM_USE_BREAKABLE_CUDAGRAPH | `0` (serve.env) | `0` | ok |
| env VLLM_PLUGINS | `dsv4_vision` (run.sh literal) | `dsv4_vision` | ok |
| env NCCL_IB_HCA | `rocep1s0f1` (serve.env HCA) | `rocep1s0f1` | ok |
| env NCCL_SOCKET_IFNAME | `enp1s0f1np1` (serve.env IFACE) | `enp1s0f1np1` | ok |
| env VLLM_USE_B12X_MOE/FP8_GEMM/WO_PROJECTION/SPARSE_INDEXER/V2_MODEL_RUNNER | `1,1,1,1,1` (run.sh literal) | `1,1,1,1,1` | ok |
| env B12X_MLA_SM120_UNIFIED,B12X_MOE_FORCE_A8 | `1,1` (run.sh literal) | `1,1` | ok |
| HF cache mount | `/home/sfxnz/.cache/huggingface -> /cache/huggingface` (serve.env HF_CACHE) | `/home/sfxnz/.cache/huggingface -> /cache/huggingface` | ok |
| host network / host ipc / shm 32g | `host/host/34359738368` (run.sh literal) | `host/host/34359738368` | ok |
| rank args | `--headless, --node-rank 1` (run.sh) | `--headless, --node-rank 1` | ok |
| EXTRA_ARGS (flags not emitted by run.sh) | `none` (serve.env EXTRA_ARGS="") | `none` | ok |

Full argv: `["/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/86f746b36186f0e567729a5c06a8c918caba82a9", "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", "1", "--distributed-executor-backend", "mp", "--master-addr", "10.100.8.1", "--master-port", "29522", "--headless", "--max-model-len", "1048576", "--kv-cache-dtype", "fp8", "--kv-cache-memory", "12884901888", "--gpu-memory-utilization", "0.80", "--max-num-seqs", "2", "--max-num-batched-tokens", "8192", "--compilation-config", "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}", "--max-cudagraph-capture-size", "16", "--block-size", "256", "--moe-backend", "b12x", "--linear-backend", "b12x", "--attention-backend", "B12X_MLA_SPARSE", "--speculative-config", "{\"method\":\"dspark\",\"num_speculative_tokens\":6,\"draft_sample_method\":\"probabilistic\",\"attention_backend\":\"B12X_MLA_SPARSE\"}", "--tokenizer-mode", "deepseek_v4", "--tool-call-parser", "deepseek_v4", "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v4", "--reasoning-config", "{\"reasoning_parser\":\"deepseek_v4\",\"reasoning_start_str\":\"\",\"reasoning_end_str\":\"\"}", "--default-chat-template-kwargs", "{\"thinking\":false,\"reasoning_effort\":\"low\"}", "--limit-mm-per-prompt", "{\"image\":8}", "--enable-prefix-caching", "--served-model-name", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "--trust-remote-code"]`


## Verdict

MATCH: every serve-relevant value equals the recipe default on both ranks. Benching in place as the recipe number.


Notes: rank 1 has no HF_TOKEN env (worker `token_env` found no token file on spark2); not serve-relevant, weights load from the mounted snapshot. Image ids differ between nodes because `dsv4-flash-vision-sm121` is built locally on each node (RepoDigests empty); each container runs the current local tag on its node.

## Image content vs repo HEAD (caveat, not a flag mismatch)

`recipe.yaml` has `image.digest: null`; the recipe's image is whatever local `dsv4-flash-vision-sm121` a node has built. Both containers run the current local tag on their node (image id check above). But the local images were built 2026-09-01 13:44 BST, before PRs #1/#2/#3 merged (14:42–14:52 BST), and their plugin sources do not match `docker/plugin/dsv4_vision` at HEAD a34220f. Hashes: `docker-tree-repo.sha256`, `docker-tree-rank0.sha256`, `docker-tree-rank1.sha256`; diffs: `image-drift.diff`.

- rank 0 (spark1): only `encoding_dsv4.py` differs (HEAD adds `flatten_content_blocks` and image-block rendering from #3). `hooks.py`, `vl_model.py`, `mm_preprocess.py`, `model.py`, `__init__.py` match HEAD (the #2 multimodal registration).
- rank 1 (spark2): older plugin build. `__init__.py` still registers `DeepseekV4VisionForCausalLM` under `DeepseekV4ForCausalLM`; `model.py` is the pre-#2 wrapper; `vision.py` and `encoding_dsv4.py` differ; `hooks.py`, `mm_preprocess.py`, `vl_model.py` are absent. spark2's image was never rebuilt after #2.

Both plugin versions subclass the B12X `DeepseekV4ForCausalLM` for the language path and skip `bias_vl`, and the TP=2 serve has been up 32 h, so text decode runs the same engine path on both ranks. `./run.sh` does not build images, so a fresh boot would start these same images; rebuilding while holding the lease is forbidden by the brief. Per the user's decision the running serve is benched in place. Treat the vision probe result as "rank 0 plugin at #2, rank 1 plugin pre-#2". Rebuild `dsv4-flash-vision-sm121` on both nodes from HEAD before the next bench if the image is meant to track the repo.
