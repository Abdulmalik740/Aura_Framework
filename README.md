# AURA: A Config-Driven Framework for Brand Voice Control in Customer Service LLMs

AURA (Adaptive Unified Response Architecture) is a modular, configuration-driven framework for enforcing brand voice consistency in LLM-based customer service systems. All brand identity — tone profiles, forbidden phrases, behavioral rules, linguistic constraints — is encoded in a single declarative YAML file. No brand-specific logic is hardcoded in the Python codebase. Switching brands requires only swapping the config file.

The repository includes two brand instantiations:

- **Apex Stride** — performance running brand (short imperatives, high contraction, direct accountability)
- **Ember & Edge** — craft kitchenware brand (sensory language, craft narrative, poetic tone)

## Quickstart

### 1. Install dependencies

```bash
pip install openai pyyaml flask
```

### 2. Set API key

```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Run Apex Stride

```bash
cd apex_stride
python web_interface_universal.py
```

### 4. Run Ember & Edge

```bash
cd ember_edge
python ember_edge_web.py
```

### 5. Run ablation evaluation

```bash
# Apex Stride
cd apex_stride && python ablation_runner.py

# Ember & Edge
cd ember_edge && python ablation_knife.py
```

---

## Brand Configuration

All brand knowledge lives in `brand_config.yaml`. A new brand requires defining:

| Section | Description |
|---|---|
| `brand` | Name, tagline, mission, personality |
| `core_values` | Values with prompt instructions |
| `messaging_pillars` | Key messages and usage guidance |
| `voice_guidelines` | Do/don't lists for tone |
| `tone_profiles` | Named profiles with numeric parameters |
| `tone_selection_rules` | Priority-ordered context-to-profile mapping |
| `brand_guard.forbidden_patterns` | Phrase blacklists with severity levels |
| `brand_guard.linguistic_dna` | Sentence length and contraction ratio thresholds |
| `brand_guard.context_rules` | Bucket-specific required/forbidden phrases |
| `llm` | Model identifiers and generation parameters |

---

## Models Used

| Role | Model |
|---|---|
| Generation | `gpt-4o-mini` |
| CUE fallback | `gpt-4o-mini` |
| Primary judge | `gpt-5.1` |
| Secondary judge | `claude-sonnet` |

---


## Core Modules

**`config_loader.py`**
Single source of truth. All modules import brand knowledge through this file. The active brand is set via the `BRAND_CONFIG_PATH` environment variable. Uses `lru_cache` so the YAML is parsed once per process.

**`context_understanding_engine.py`**
Classifies each incoming message along eight dimensions: emotion, emotion intensity, intent, urgency, user type, motivation level, escalation signals, and competitor mentions. Uses a hybrid architecture — keyword lists from YAML handle high-confidence cases in O(1); an LLM call is invoked only when rule confidence falls below a configurable threshold.

**`tone_adaptation_engine.py`**
Maps context output to one of the brand's named tone profiles using a priority-ordered rule set loaded from YAML. Applies additive context adjustments for urgency, churn risk, and user type, enforces numeric boundary constraints, and smooths abrupt tone shifts across turns.

**`brand_consistency_guard.py`**
Post-generation guardrail. Runs three independent sub-checks: forbidden phrase scan, linguistic DNA check (sentence length, contraction ratio), and bucket-specific context-tone check. If the weighted score falls below threshold θ or a critical violation is detected, it triggers targeted regeneration with a corrective prompt.

**`conversation_state_manager.py`**
Tracks session state across turns: user profile, motivation trend, risk scores (escalation, churn, competitor switch), active issues, and conversation phase. All thresholds and keyword lists come from `brand_config.yaml`.

**`instrumentation.py`**
Three lightweight trackers: AI fallback rate and per-path latency for the context engine, turn-level tone shift and smoothing rate for the tone engine, and wall-clock latency per pipeline stage.

---
