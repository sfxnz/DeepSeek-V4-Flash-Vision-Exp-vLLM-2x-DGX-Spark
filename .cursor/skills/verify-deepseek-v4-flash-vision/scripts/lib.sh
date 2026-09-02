# shellcheck shell=bash
# SC2034: these variables are read by the scripts that source this file.
# shellcheck disable=SC2034
CONTAINER_NAME="${CONTAINER_NAME:-dsv4-flash-vision-exp}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-dsv4-flash-vision-sm121}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
WORKER_HOST="${WORKER_HOST:-spark2}"
API="http://127.0.0.1:${PORT}/v1"
