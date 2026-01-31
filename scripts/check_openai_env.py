"""Quickly verify OpenRouter API connectivity using env vars.

Usage:
  python scripts/check_openai_env.py

Required env vars:
  OPENROUTER_API_KEY

Optional env vars:
  OPENROUTER_API_BASE_URL (default: https://openrouter.ai/api/v1)
  OPENROUTER_HTTP_REFERER
  OPENROUTER_APP_NAME
"""

from __future__ import annotations

import argparse
import json
import os

import requests


def _build_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def _get_base_url() -> str:
    return (
        os.getenv("OPENROUTER_API_BASE_URL")
        or os.getenv("OPENAI_API_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )


def _get_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_KEY") or ""


def _get_extra_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if app_name:
        headers["X-Title"] = app_name
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify OpenRouter connectivity and optionally validate a model name."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", "").strip(),
        help="Optional model id to validate against /models.",
    )
    args = parser.parse_args()

    base_url = _get_base_url()
    api_key = _get_api_key()

    if not api_key:
        print("Missing env var: OPENROUTER_API_KEY")
        return 2

    models_url = _build_models_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    headers.update(_get_extra_headers())

    try:
        resp = requests.get(models_url, headers=headers, timeout=15)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1

    if resp.ok:
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            payload = {"raw": resp.text[:200]}
        model_count = None
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                model_count = len(data)
        print("OK: Connected to API")
        print(f"URL: {models_url}")
        print(f"Status: {resp.status_code}")
        if model_count is not None:
            print(f"Models: {model_count}")
        if args.model:
            model_ids = set()
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "id" in item:
                            model_ids.add(item["id"])
            if args.model in model_ids:
                print(f"Model OK: {args.model}")
            else:
                print(f"Model NOT FOUND: {args.model}")
                return 3
        return 0

    print("ERROR: API responded with non-OK status")
    print(f"URL: {models_url}")
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:400]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
