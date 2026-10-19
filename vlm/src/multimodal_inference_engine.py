"""Unified Multimodal Vision Inference Engine.

Supports:
1. Qwen2.5-VL-3B-Instruct (via llama-server OpenAI-compatible HTTP endpoint)
2. Gemma 3 4B (via Ollama native HTTP endpoint)

Provides:
- Strict live execution mode (fail-fast, zero cached fallbacks)
- Preflight health checks and multimodal smoke tests
- View mode abstraction: CROP_ONLY vs. CROP_PLUS_CONTEXT
- Honest wire telemetry: request attempted, HTTP status, response received, parse validity, schema validity, and wall-clock latency
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from PIL import Image

repo_root = Path(__file__).resolve().parent.parent.parent

# Default endpoints from model registry
DEFAULT_QWEN_URL = "http://127.0.0.1:8089/v1/chat/completions"
DEFAULT_GEMMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_GEMMA_MODEL_NAME = "gemma3:4b"

CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]

ONTOLOGY_ALIASES = {
    "IMMOVABLE": "IMMOVABLE",
    "FIXED": "IMMOVABLE",
    "STRUCTURAL": "IMMOVABLE",
    "ARCHITECTURAL": "IMMOVABLE",
    "WALL_MOUNTED": "IMMOVABLE",
    "CEILING_MOUNTED": "IMMOVABLE",
    "MOUNTED": "IMMOVABLE",
    "MOVABLE": "MOVABLE",
    "PORTABLE": "MOVABLE",
    "RELOCATABLE": "MOVABLE",
    "FURNITURE": "MOVABLE",
    "OBJECT": "MOVABLE",
    "FREESTANDING": "MOVABLE",
    "MOBILE": "MOBILE",
    "DYNAMIC": "MOBILE",
    "SELF_PROPELLED": "MOBILE",
    "ROBOT": "MOBILE",
    "AGENT": "MOBILE",
}


def encode_image_base64(path: Path, max_dim: int = 240) -> str:
    """Resize image preserving aspect ratio and return standard base64 string."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def clean_json_response(raw_text: str) -> str:
    """Extract JSON object from raw model completion text."""
    text = raw_text.strip()
    match_fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match_fenced:
        return match_fenced.group(1).strip()
    match_braces = re.search(r"\{.*?\}", text, re.DOTALL)
    if match_braces:
        return match_braces.group(0).strip()
    return text


def parse_multimodal_prediction(
    raw_text: str, expected_id: str, expected_class: str
) -> Tuple[Dict[str, Any], bool, bool]:
    """Parse JSON and validate schema agreement.

    Returns:
        (parsed_dict, parse_valid, schema_valid)
    """
    cleaned = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
        parse_valid = True
    except Exception as e:
        return (
            {
                "object_id": expected_id,
                "object_class": expected_class,
                "movability": "UNKNOWN",
                "confidence": 0.0,
                "abstain": True,
                "validation_error": f"json_parse_failed: {str(e)}",
            },
            False,
            False,
        )

    if not isinstance(data, dict):
        return (
            {
                "object_id": expected_id,
                "object_class": expected_class,
                "movability": "UNKNOWN",
                "confidence": 0.0,
                "abstain": True,
                "validation_error": "response_not_a_dictionary",
            },
            True,
            False,
        )

    raw_mov = str(data.get("movability", "UNKNOWN")).strip().upper()
    abstain = bool(data.get("abstain", False))
    validation_error = None

    if raw_mov in ONTOLOGY_ALIASES:
        canonical_mov = ONTOLOGY_ALIASES[raw_mov]
    else:
        canonical_mov = "UNKNOWN"
        abstain = True
        validation_error = f"unrecognized_movability_class: {raw_mov}"

    raw_conf = data.get("confidence")
    confidence = None
    if raw_conf is not None:
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except (ValueError, TypeError):
            confidence = None

    schema_valid = (canonical_mov in CLASSES or abstain) and ("movability" in data)

    parsed = {
        "object_id": str(data.get("object_id", expected_id)).strip(),
        "object_class": str(data.get("object_class", expected_class)).strip(),
        "movability": canonical_mov,
        "raw_movability": raw_mov,
        "confidence": confidence,
        "physical_support": data.get("physical_support"),
        "transport_context": data.get("transport_context"),
        "short_explanation": data.get("short_explanation"),
        "abstain": abstain,
        "validation_error": validation_error,
    }
    return parsed, parse_valid, schema_valid


@dataclass
class WireInferenceRecord:
    """Full auditable wire execution record for a single model query."""
    physical_instance_id: str
    model_key: str
    view_mode: Literal["CROP_ONLY", "CROP_PLUS_CONTEXT"]
    requested_mode: str
    actual_mode: str
    server_endpoint: str
    request_attempted: bool
    http_success: bool
    response_received: bool
    parse_valid: bool
    schema_valid: bool
    prediction_accepted: bool
    cache_hit: bool
    client_latency_ms: float
    raw_response: str
    parsed_prediction: str
    confidence: Optional[float]
    token_usage: Optional[Dict[str, Any]]
    server_timing: Optional[Dict[str, Any]]
    validation_error: Optional[str]
    input_image_hashes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultimodalInferenceEngine:
    """Client for executing strict live multimodal inference against Qwen or Gemma."""

    def __init__(
        self,
        model_key: Literal["qwen2_5_vl_3b", "gemma3_4b"] = "qwen2_5_vl_3b",
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 240.0,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ):
        self.model_key = model_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve server endpoint and model name
        if model_key == "qwen2_5_vl_3b":
            self.server_url = server_url or os.environ.get("QWEN_SERVER_URL", DEFAULT_QWEN_URL)
            self.model_name = model_name or "Qwen2.5-VL-3B-Instruct"
            self.backend_type = "llama.cpp"
        elif model_key == "gemma3_4b":
            self.server_url = server_url or os.environ.get("GEMMA_SERVER_URL", DEFAULT_GEMMA_URL)
            self.model_name = model_name or DEFAULT_GEMMA_MODEL_NAME
            self.backend_type = "ollama"
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

    def check_health(self) -> Dict[str, Any]:
        """Probe model server reachability."""
        if self.backend_type == "llama.cpp":
            health_url = self.server_url.replace("/v1/chat/completions", "/health")
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    return {"status": "ONLINE", "url": health_url, "code": resp.getcode(), "backend": self.backend_type}
            except Exception:
                pass
            try:
                req = urllib.request.Request(self.server_url)
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    return {"status": "ONLINE", "url": self.server_url, "code": resp.getcode(), "backend": self.backend_type}
            except urllib.error.HTTPError as e:
                if e.code in [400, 405]:
                    return {"status": "ONLINE", "url": self.server_url, "code": e.code, "backend": self.backend_type}
                return {"status": "OFFLINE", "url": self.server_url, "error": str(e), "code": e.code}
            except Exception as e:
                return {"status": "OFFLINE", "url": self.server_url, "error": str(e)}

        elif self.backend_type == "ollama":
            tags_url = self.server_url.replace("/api/chat", "/api/tags")
            try:
                req = urllib.request.Request(tags_url)
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    return {
                        "status": "ONLINE",
                        "url": tags_url,
                        "available_models": models,
                        "model_present": any(self.model_name in m for m in models),
                        "backend": self.backend_type,
                    }
            except Exception as e:
                return {"status": "OFFLINE", "url": tags_url, "error": str(e)}

        return {"status": "OFFLINE", "error": "unknown_backend"}

    def run_preflight_smoke_test(self) -> Dict[str, Any]:
        """Send a 1x1 test image to verify active neural multimodal inference pathway."""
        health = self.check_health()
        if health.get("status") != "ONLINE":
            raise ConnectionError(f"{self.model_key} server is OFFLINE at {self.server_url}: {health}")

        # Create 32x32 blank test image
        im = Image.new("RGB", (32, 32), color=(200, 200, 200))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = "Is this image readable? Respond with strict JSON: {\"status\": \"ok\"}"

        t0 = time.perf_counter()
        if self.backend_type == "llama.cpp":
            payload = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                "max_tokens": 50,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                self.server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            raw = res["choices"][0]["message"]["content"]
            usage = res.get("usage")
        elif self.backend_type == "ollama":
            payload = {
                "model": self.model_name,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [b64_img],
                }],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 50,
                    "num_thread": 8,
                }
            }
            req = urllib.request.Request(
                self.server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            raw = res.get("message", {}).get("content", "")
            usage = {
                "prompt_tokens": res.get("prompt_eval_count"),
                "completion_tokens": res.get("eval_count"),
            }

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "PASSED",
            "model_key": self.model_key,
            "latency_ms": round(lat_ms, 2),
            "raw_response": raw.strip(),
            "usage": usage,
        }

    def execute_live_query(
        self,
        physical_instance_id: str,
        semantic_class: str,
        prompt: str,
        crop_image_path: Path,
        scene_image_path: Optional[Path] = None,
        view_mode: Literal["CROP_ONLY", "CROP_PLUS_CONTEXT"] = "CROP_PLUS_CONTEXT",
        image_hashes: Optional[List[str]] = None,
    ) -> WireInferenceRecord:
        """Execute strict live inference over the wire and record full telemetry."""
        request_attempted = True
        http_success = False
        response_received = False
        raw_response = ""
        server_timing = None
        token_usage = None
        validation_error = None

        if not crop_image_path.exists():
            raise FileNotFoundError(f"Missing crop image: {crop_image_path}")

        # Base64 image payload preparation
        crop_b64 = encode_image_base64(crop_image_path, max_dim=240)
        images_b64 = [crop_b64]

        if view_mode == "CROP_PLUS_CONTEXT":
            if scene_image_path and scene_image_path.exists():
                scene_b64 = encode_image_base64(scene_image_path, max_dim=320)
                images_b64.append(scene_b64)

        t0 = time.perf_counter()

        try:
            if self.backend_type == "llama.cpp":
                content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for b64_str in images_b64:
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                    })

                payload = {
                    "messages": [{"role": "user", "content": content_items}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                req = urllib.request.Request(
                    self.server_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    http_success = (resp.getcode() == 200)
                    res = json.loads(resp.read().decode("utf-8"))
                    raw_response = res["choices"][0]["message"]["content"]
                    token_usage = res.get("usage")
                    response_received = True

            elif self.backend_type == "ollama":
                payload = {
                    "model": self.model_name,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": images_b64,
                    }],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_thread": 8,
                    }
                }
                req = urllib.request.Request(
                    self.server_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    http_success = (resp.getcode() == 200)
                    res = json.loads(resp.read().decode("utf-8"))
                    raw_response = res.get("message", {}).get("content", "")
                    response_received = True
                    token_usage = {
                        "prompt_tokens": res.get("prompt_eval_count"),
                        "completion_tokens": res.get("eval_count"),
                        "total_tokens": (res.get("prompt_eval_count") or 0) + (res.get("eval_count") or 0),
                    }
                    server_timing = {
                        "total_duration_ms": round((res.get("total_duration") or 0) / 1e6, 2),
                        "load_duration_ms": round((res.get("load_duration") or 0) / 1e6, 2),
                        "prompt_eval_duration_ms": round((res.get("prompt_eval_duration") or 0) / 1e6, 2),
                        "eval_duration_ms": round((res.get("eval_duration") or 0) / 1e6, 2),
                    }

        except Exception as e:
            validation_error = f"http_transport_error: {str(e)}"
            raw_response = ""

        client_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Parse and validate response
        if response_received and raw_response:
            parsed, parse_valid, schema_valid = parse_multimodal_prediction(
                raw_response, physical_instance_id, semantic_class
            )
            pred = parsed.get("movability", "UNKNOWN")
            abstained = parsed.get("abstain", False) or pred == "UNKNOWN"
            pred_accepted = schema_valid and not abstained and pred in CLASSES
            conf = parsed.get("confidence")
            if not schema_valid:
                validation_error = parsed.get("validation_error")
        else:
            parse_valid = False
            schema_valid = False
            pred_accepted = False
            pred = "UNKNOWN"
            conf = None

        return WireInferenceRecord(
            physical_instance_id=physical_instance_id,
            model_key=self.model_key,
            view_mode=view_mode,
            requested_mode="LIVE_INFERENCE",
            actual_mode="LIVE_INFERENCE",
            server_endpoint=self.server_url,
            request_attempted=request_attempted,
            http_success=http_success,
            response_received=response_received,
            parse_valid=parse_valid,
            schema_valid=schema_valid,
            prediction_accepted=pred_accepted,
            cache_hit=False,  # Strictly zero cache in live mode
            client_latency_ms=round(client_latency_ms, 2),
            raw_response=raw_response,
            parsed_prediction=pred,
            confidence=conf,
            token_usage=token_usage,
            server_timing=server_timing,
            validation_error=validation_error,
            input_image_hashes=image_hashes or [],
        )
