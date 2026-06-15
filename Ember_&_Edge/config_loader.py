"""
config_loader.py
Single source of truth. Every module imports from here.
Usage:
    from config_loader import cfg, brand, products, tone_profiles, llm
"""

import yaml
import os
from pathlib import Path
from functools import lru_cache
from typing import Any

_CONFIG_PATH = Path(os.getenv("BRAND_CONFIG_PATH", Path(__file__).parent / "brand_config.yaml"))


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache config. Override path via BRAND_CONFIG_PATH env var."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg(*keys: str, default: Any = None) -> Any:
    """
    Dot-path accessor into config.
    Example: cfg("brand", "name")  →  "Apex Stride"
             cfg("llm", "generation_model")  →  "gpt-4o-mini"
    """
    data = load_config()
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is None:
            return default
    return data


# ── Convenience top-level accessors ──────────────────────────────────────────

def brand() -> dict:
    return cfg("brand")

def products() -> dict:
    return cfg("products")

def tone_profiles() -> dict:
    return cfg("tone_profiles")

def llm() -> dict:
    return cfg("llm")

def brand_guard_config() -> dict:
    return cfg("brand_guard")

def context_engine_config() -> dict:
    return cfg("context_engine")

def conversation_state_config() -> dict:
    return cfg("conversation_state")

def recommendation_rules() -> dict:
    return cfg("recommendation_rules")

def tone_selection_rules() -> list:
    return cfg("tone_selection_rules")

def tone_adjustments() -> dict:
    return cfg("tone_adjustments")

def tone_brand_boundaries() -> dict:
    return cfg("tone_brand_boundaries")

def tone_vocabulary() -> dict:
    return cfg("tone_vocabulary_instructions")

def special_rules() -> dict:
    return cfg("special_rules")

def demo_scenarios() -> list:
    return cfg("demo_scenarios")

def build_brand_ethos() -> dict:
    """Backwards-compatible dict matching the old APEX_STRIDE_ETHOS shape."""
    c = load_config()
    return {
        "name": c["brand"]["name"],
        "tagline": c["brand"]["tagline"],
        "mission": c["brand"]["mission"],
        "personality": c["brand"]["personality"],
        "core_values": [v["name"] for v in c["core_values"]],
        "core_values_detail": c["core_values"],
        "messaging_pillars": [p["text"] for p in c["messaging_pillars"]],
        "messaging_pillars_detail": c["messaging_pillars"],
        "voice_guidelines": c["voice_guidelines"],
    }
