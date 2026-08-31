#!/usr/bin/env bash
# DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
IMAGE="${IMAGE:-dsv4-flash-vision-sm121}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv4-flash-vision-exp}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29522}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1}"
TP="${TP:-2}"
NNODES="${NNODES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-327680}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
UTIL="${UTIL:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-4445787956}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
SPEC="${SPEC:-dspark}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
FORCE_UNSAFE_CTX="${FORCE_UNSAFE_CTX:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-86f746b36186f0e567729a5c06a8c918caba82a9}"
SNAPSHOT="${HF_CACHE}/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/${SNAPSHOT_SHA}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORBIDDEN_CONTAINER="glm53-flash-nvfp4"

if [[ -z "${SPEC_CONFIG:-}" ]]; then
  case "$SPEC" in
    dspark)
      SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"',"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}'
      ;;
    *)
      echo "Unknown SPEC=$SPEC (want dspark)" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
fi

if [[ "$KV_CACHE_DTYPE" == fp8 && "$MAX_MODEL_LEN" -gt 327680 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "fp8 KV pin cannot hold --max-model-len $MAX_MODEL_LEN. A 1M window needs more UMA than GB10 survives. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$KV_CACHE_DTYPE" == fp8_e4m3 && "$MAX_MODEL_LEN" -gt 327680 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "fp8 KV pin cannot hold --max-model-len $MAX_MODEL_LEN. A 1M window needs more UMA than GB10 survives. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s eager=%s compilation=%s image=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$ENFORCE_EAGER" "$COMPILATION_CONFIG" "$IMAGE"
  exit 0
fi

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

token_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    tr -d '[:space:]' <"$HOME/.cache/huggingface/token"
  fi
}

resolve_model() {
  if [[ -d "$SNAPSHOT" ]]; then
    printf '%s\n' "$SNAPSHOT_IN_CONTAINER"
  else
    printf '%s\n' "$MODEL"
  fi
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  fi
}

ensure_image() {
  log "Ensuring image $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build docker/Dockerfile.b12x-vision from eugr/spark-vllm-b12x:latest first. Do not use stock vllm/vllm-openai on sm_121." >&2
    exit 1
  fi
}

ensure_weights() {
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    return
  fi
  local HF=""
  HF="$(hf_bin || true)"
  if [[ -d "$SNAPSHOT" ]]; then
    log "Using pinned snapshot $SNAPSHOT"
  elif [[ -n "$HF" ]]; then
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    log "Downloading $MODEL (resumes under $HF_CACHE)"
    "$HF" download "$MODEL"
  else
    log "No hf CLI on PATH — vLLM will pull weights on first load"
  fi
}

refuse_foreign_serve() {
  if docker ps --format '{{.Names}}' | grep -qx "$FORBIDDEN_CONTAINER"; then
    echo "$FORBIDDEN_CONTAINER is running. This recipe needs exclusive GPUs on both Sparks. Do not start. Do not docker rm that container from this script." >&2
    exit 1
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE"
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found" >&2
    exit 1
  fi
  refuse_foreign_serve
  maybe_drop_caches
  stop_local
  ensure_image
  ensure_weights

  local serve_model
  serve_model="$(resolve_model)"

  local tok
  tok="$(token_env || true)"
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "VLLM_PLUGINS=dsv4_vision"
    -e "CUTE_DSL_ARCH=sm_121a"
    -e "VLLM_USE_AOT_COMPILE=1"
    -e "VLLM_USE_BREAKABLE_CUDAGRAPH=$VLLM_USE_BREAKABLE_CUDAGRAPH"
    -e "VLLM_USE_MEGA_AOT_ARTIFACT=-1"
    -e "VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1"
    -e "VLLM_USE_FLASHINFER_SAMPLER=1"
    -e "VLLM_USE_B12X_WO_PROJECTION=1"
    -e "VLLM_USE_B12X_MHC=1"
    -e "VLLM_USE_B12X_FP8_GEMM=1"
    -e "VLLM_USE_B12X_MOE=1"
    -e "VLLM_USE_B12X_SPARSE_INDEXER=1"
    -e "VLLM_USE_V2_MODEL_RUNNER=1"
    -e "B12X_MLA_SM120_UNIFIED=1"
    -e "B12X_MOE_FORCE_A8=1"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "GLOO_SOCKET_IFNAME=$IFACE"
    -e "TP_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_NVLS_ENABLE=0"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=WARN"
  )
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")
  if [[ -n "$tok" ]]; then
    env_args+=(-e "HF_TOKEN=$tok" -e "HUGGING_FACE_HUB_TOKEN=$tok")
  fi

  local rank_args=()
  if [[ "$rank" == "0" ]]; then
    rank_args+=(--host 0.0.0.0 --port "$PORT")
  else
    rank_args+=(--headless)
  fi

  local eager_args=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  else
    eager_args+=(--compilation-config "$COMPILATION_CONFIG" --max-cudagraph-capture-size 16)
  fi

  local vol_args=(-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}")
  local batched_args=()
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    batched_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  local load_args=()
  if [[ "$LOAD_FORMAT" != "auto" ]]; then
    load_args+=(--load-format "$LOAD_FORMAT")
  fi
  local kv_dtype_arg="$KV_CACHE_DTYPE"
  if [[ "$kv_dtype_arg" == fp8_e4m3 ]]; then
    kv_dtype_arg=fp8
  fi

  log "Starting $CONTAINER_NAME rank=$rank model=$serve_model ctx=$MAX_MODEL_LEN kv=$KV_CACHE_MEMORY spec=$SPEC"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart no \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    "${vol_args[@]}" \
    "${env_args[@]}" \
    "$IMAGE" \
    "$serve_model" \
    --tensor-parallel-size "$TP" \
    --nnodes "$NNODES" \
    --node-rank "$rank" \
    --distributed-executor-backend mp \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    "${rank_args[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$kv_dtype_arg" \
    --kv-cache-memory "$KV_CACHE_MEMORY" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${batched_args[@]}" \
    "${eager_args[@]}" \
    --block-size "$BLOCK_SIZE" \
    --moe-backend "$MOE_BACKEND" \
    --linear-backend b12x \
    --attention-backend B12X_MLA_SPARSE \
    --speculative-config "$SPEC_CONFIG" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}' \
    --limit-mm-per-prompt '{"image":8}' \
    --enable-prefix-caching \
    "${load_args[@]}" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    $EXTRA_ARGS
}

wait_ready() {
  log "Waiting for http://127.0.0.1:${PORT}/v1/models"
  local i
  for i in $(seq 1 480); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      log "Ready → http://127.0.0.1:${PORT}/v1  (context=$MAX_MODEL_LEN)"
      curl -s "http://127.0.0.1:${PORT}/v1/models" || true
      echo
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  if command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
    log "Starting worker on $WORKER_HOST first"
    scp -q "$0" "${WORKER_HOST}:/tmp/dsv4-vision-run.sh"
    ssh "$WORKER_HOST" \
      "ROLE=worker ORCHESTRATE=0 IMAGE='$IMAGE' CONTAINER_NAME='$CONTAINER_NAME' PORT='$PORT' MASTER_PORT='$MASTER_PORT' HEAD_IP='$HEAD_IP' IFACE='$IFACE' HCA='$HCA' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' UTIL='$UTIL' KV_CACHE_MEMORY='$KV_CACHE_MEMORY' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' BLOCK_SIZE='$BLOCK_SIZE' TP='$TP' NNODES='$NNODES' SERVED_NAME='$SERVED_NAME' SKIP_DOWNLOAD='$SKIP_DOWNLOAD' SPEC='$SPEC' SPEC_CONFIG='$SPEC_CONFIG' NUM_SPECULATIVE_TOKENS='$NUM_SPECULATIVE_TOKENS' ENFORCE_EAGER='$ENFORCE_EAGER' COMPILATION_CONFIG='$COMPILATION_CONFIG' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' FORCE_UNSAFE_CTX='$FORCE_UNSAFE_CTX' VLLM_USE_BREAKABLE_CUDAGRAPH='$VLLM_USE_BREAKABLE_CUDAGRAPH' LOAD_FORMAT='$LOAD_FORMAT' MOE_BACKEND='$MOE_BACKEND' EXTRA_ARGS='$EXTRA_ARGS' bash /tmp/dsv4-vision-run.sh"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
  else
    log "Cannot SSH to $WORKER_HOST — starting local rank only. Run ROLE=worker ./run.sh on the other Spark first."
  fi
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
fi
