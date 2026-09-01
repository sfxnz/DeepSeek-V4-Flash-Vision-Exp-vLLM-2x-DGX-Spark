#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
body="$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"thinking": false, "reasoning_effort": "low"}
  }')"
if [[ "$body" != *'"choices"'* || "$body" != *'"content"'* ]]; then
  echo "smoke: response missing choices/content" >&2
  printf '%s\n' "$body" >&2
  exit 1
fi
printf '%s\n' "$body"
