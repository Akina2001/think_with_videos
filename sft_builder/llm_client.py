from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config import APIConfig
from .text_utils import ensure_dir, parse_json_response, sha1_file, sha1_text, write_json


class OpenAICompatibleVideoClient:
    def __init__(self, api_config: APIConfig, cache_dir: str | Path) -> None:
        self.api_config = api_config
        self.cache_dir = ensure_dir(cache_dir)
        self.session = requests.Session()

        api_key = api_config.api_key
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_config.api_key} is not set."
            )
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _guess_mime_type(video_path: str) -> str:
        ext = Path(video_path).suffix.lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
        }
        return mime_map.get(ext, "video/mp4")

    def _encode_video_to_data_url(self, video_path: str) -> str:
        mime_type = self._guess_mime_type(video_path)
        with open(video_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _make_cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[str],
        mm_processor_kwargs: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "url": self.api_config.chat_completions_url,
            "model": self.api_config.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "video_hash": sha1_file(video_path) if video_path else None,
            "mm_processor_kwargs": mm_processor_kwargs or {},
            "temperature": self.api_config.temperature,
            "top_p": self.api_config.top_p,
            "max_tokens": self.api_config.max_tokens,
        }
        return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _extract_text_content(self, response_json: Dict[str, Any]) -> str:
        content = response_json["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    out.append(item["text"])
                else:
                    out.append(str(item))
            return "\n".join(out)
        return str(content)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[str] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        force_json: bool = True,
    ) -> Any:
        mm_kwargs = dict(self.api_config.default_mm_processor_kwargs)
        if mm_processor_kwargs:
            mm_kwargs.update(mm_processor_kwargs)

        cache_key = self._make_cache_key(system_prompt, user_prompt, video_path, mm_kwargs)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            raw_text = cached["raw_text"]
            return parse_json_response(raw_text) if force_json else raw_text

        content = [{"type": "text", "text": user_prompt}]
        if video_path is not None:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": self._encode_video_to_data_url(video_path)
                    },
                }
            )

        payload = {
            "model": self.api_config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.api_config.temperature,
            "top_p": self.api_config.top_p,
            "max_tokens": self.api_config.max_tokens,
            "stream": False,
        }
        # if video_path is not None:
        #     payload["mm_processor_kwargs"] = mm_kwargs

        response = self.session.post(
            self.api_config.chat_completions_url,
            headers=self._headers(),
            json=payload,
            timeout=self.api_config.timeout,
        )
        response.raise_for_status()
        response_json = response.json()
        raw_text = self._extract_text_content(response_json)

        write_json(
            cache_path,
            {
                "request_payload": payload,
                "response_json": response_json,
                "raw_text": raw_text,
            },
        )
        return parse_json_response(raw_text) if force_json else raw_text
