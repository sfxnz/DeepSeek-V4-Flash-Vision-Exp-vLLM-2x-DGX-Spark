#!/usr/bin/env python3
"""Live OpenAI-compat vision smoke against a running serve."""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request


def _red_jpeg() -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("PIL is required for smoke_vision.py") from exc
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 60)).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
    args = parser.parse_args()
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is this image? Reply with one word only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_red_jpeg()}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        err = exc.read().decode()
        print(err, file=sys.stderr)
        if "is not a multimodal model" in err:
            print("result=fail reason=not_multimodal", file=sys.stderr)
        return 1
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning") or "")
    print(json.dumps({"content": msg.get("content"), "finish_reason": data["choices"][0].get("finish_reason")}, indent=2))
    if "is not a multimodal model" in text:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
