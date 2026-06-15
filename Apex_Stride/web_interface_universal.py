"""
Web Interface - Brand-agnostic
All brand names, models, pricing, sample queries, prompts, and
gratitude responses come from brand_config.yaml.
"""

from flask import Flask, request, jsonify
import json
import secrets
import random
from datetime import datetime
import time
import re
import os

from universal_adaptive_framework import UniversalAdaptiveFramework, client
from config_loader import cfg, build_brand_ethos

# ── Load all config up front ──────────────────────────────────────────────────
_brand   = build_brand_ethos()
_web     = cfg("web")
_BRAND_NAME    = _brand["name"]
_BRAND_TAGLINE = _brand["tagline"]
_BRAND_PILLAR  = _brand["messaging_pillars"][0]  
_EVAL_CFG      = _web["evaluation_criteria"]

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

framework = UniversalAdaptiveFramework(client)


# ── Baseline system ───────────────────────────────────────────────────────────
class BaselineSystem:
    def __init__(self, brand_ethos=None):
        self.client = client
        self._web_cfg = cfg("web")
        self._model = self._web_cfg["baseline_model"]
        self._temp  = self._web_cfg["baseline_temperature"]
        self._max   = self._web_cfg["baseline_max_tokens"]

    def generate_response(self, message, conversation_history=None):
        step_start = time.time()

        is_first_message = not conversation_history or len(conversation_history) == 0
        greeting_rule = (
            self._web_cfg["baseline_greeting_first"]
            if is_first_message
            else self._web_cfg["baseline_greeting_subsequent"]
        )

        system_prompt = self._web_cfg["baseline_system_prompt"].format(
            greeting_rule=greeting_rule
        )

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for exchange in conversation_history[-3:]:
                if 'user' in exchange and 'assistant' in exchange:
                    messages.append({"role": "user", "content": exchange['user']})
                    messages.append({"role": "assistant", "content": exchange['assistant']})

        messages.append({"role": "user", "content": message})
        response = client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=self._temp, max_tokens=self._max,
        )
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return {
            "response": response.choices[0].message.content,
            "model": self._model,
            "tokens": tokens,
            "time": time.time() - step_start,
        }

baseline_system = BaselineSystem()


# ── Brand evaluator ───────────────────────────────────────────────────────────
class BrandEvaluator:
    def __init__(self, brand_doc_path=None):
        self.brand_doc_path    = brand_doc_path
        self.brand_doc_content = None
        self._eval_model  = _web["evaluator_model"]
        self._eval_temp   = _web["evaluator_temperature"]
        self._eval_max    = _web["evaluator_max_tokens"]
        self._cmp_model   = _web["comparison_model"]
        self._cmp_temp    = _web["comparison_temperature"]
        self._cmp_max     = _web["comparison_max_tokens"]

    def _extract_json(self, text):
        if not text or not text.strip():
            return "{}"
        for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```", r"\{.*\}"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1) if m.lastindex else m.group(0)
        return text.strip()

    def evaluate_response_blind(self, chat_content, response, context):
        eval_start = time.time()
        if self.brand_doc_content:
            messages = self._build_eval_messages(chat_content, response, context)
        else:
            messages = self._build_minimal_eval_messages(chat_content, response, context)

        try:
            resp_obj = client.chat.completions.create(
                model=self._eval_model, messages=messages,
                temperature=self._eval_temp,
                max_completion_tokens=self._eval_max,
                response_format={"type": "json_object"},
            )
            raw = resp_obj.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty evaluator response")

            result = json.loads(self._extract_json(raw))

            if "scores" not in result:
                result["scores"] = {"Brand_Voice_Consistency": 3, "Contextual_Intelligence": 3, "Tone_Adaptation_Within_Brand": 3}
            if not result.get("weighted_score"):
                result["weighted_score"] = round(sum(result["scores"].values()) / 3, 2)
            result.setdefault("strengths", ["Not provided"])
            result.setdefault("weaknesses", ["Not provided"])

            tokens = 0
            if hasattr(resp_obj, "usage") and resp_obj.usage:
                tokens = getattr(resp_obj.usage, "total_tokens", 0) or \
                         getattr(resp_obj.usage, "completion_tokens", 0) + getattr(resp_obj.usage, "prompt_tokens", 0)
            if tokens == 0:
                tokens = int(200 + (len(self.brand_doc_content.split()) if self.brand_doc_content else 0))

            eval_time = time.time() - eval_start
            result["_meta"] = {"model": self._eval_model, "tokens": tokens, "time": eval_time}
            return result

        except Exception as e:
            print(f"Evaluation error: {e}")
            eval_time = time.time() - eval_start
            return {
                "scores": {"Brand_Voice_Consistency": 3, "Contextual_Intelligence": 3, "Tone_Adaptation_Within_Brand": 3},
                "weighted_score": 3.0,
                "strengths": ["Unable to evaluate"],
                "weaknesses": ["Evaluation failed"],
                "_meta": {"model": "N/A", "tokens": 0, "time": eval_time},
            }

    def _build_eval_messages(self, chat_content, response, context):
        criteria = _EVAL_CFG
        score5_lines  = "\n".join(f"   ✓ {l}" for l in criteria["score_5_voice"])
        score1_lines  = "\n".join(f"   ✗ {l}" for l in criteria["score_1_voice"])
        system_rules  = "\n".join(f"{i+1}. {r}" for i, r in enumerate(criteria["evaluator_system_rules"]))

        prompt = f"""You are evaluating BRAND ALIGNMENT for {_BRAND_NAME}.

CRITICAL CONTEXT:
{_BRAND_NAME} is NOT a traditional customer service brand. Their ethos is:
{criteria["brand_personality_note"]}

RESPONSE TO EVALUATE:
"{response}"

USER MESSAGE: "{chat_content}"

=== EVALUATION CRITERIA ===

1. BRAND_VOICE_CONSISTENCY (Is this how {_BRAND_NAME} sounds?)

   SCORE 5 - Perfect {_BRAND_NAME} voice:
{score5_lines}

   SCORE 1 - Generic corporate voice:
{score1_lines}


2. CONTEXTUAL_INTELLIGENCE (Does it understand what {_BRAND_NAME} customers need?)

   SCORE 5: Recognizes customer needs ACTION not explanation. Cuts through quickly. Motivates through challenge.
   SCORE 3: Helpful but too much hand-holding. Over-explains.
   SCORE 1: Lengthy explanations. Excessive caveats. Talks AT customer.

   IMPORTANT: Brevity + insight = HIGH score. Length + detail = LOW score.

3. TONE_ADAPTATION_WITHIN_BRAND (Right {_BRAND_NAME} voice for this specific situation?)

   SCORE 5: Maintains brand voice AND fits the moment. Adjusts intensity appropriately.
   SCORE 3: Maintains brand but misses context, OR adapts but loses brand.
   SCORE 1: Wrong tone for situation AND off-brand.

Return ONLY this JSON:
{{
  "scores": {{
    "Brand_Voice_Consistency": X,
    "Contextual_Intelligence": X,
    "Tone_Adaptation_Within_Brand": X
  }},
  "weighted_score": X.X,
  "strengths": ["Specific quote showing what worked"],
  "weaknesses": ["Specific quote showing what didn't work"]
}}"""

        return [
            {"role": "system", "content": f"""You are a BRAND COMPLIANCE evaluator for {_BRAND_NAME}.

CRITICAL RULES:
{system_rules}

Be STRICT. Most traditional "good customer service" responses should score 1-2 on Brand_Voice.
Return ONLY valid JSON."""},
            {"role": "user", "content": prompt},
        ]

    def _build_minimal_eval_messages(self, chat_content, response, context):
        voice_desc = _EVAL_CFG["fallback_eval_voice_description"]
        markers    = _EVAL_CFG["fallback_eval_markers"]
        prompt = f"""Evaluate this response against {_BRAND_NAME} brand.

USER: "{chat_content}"
CONTEXT: {context.get("intent")} / {context.get("emotion")}

RESPONSE: "{response}"

{_BRAND_NAME.upper()} BRAND:
- Voice: {voice_desc}
- Markers: {markers}

Rate 1-5 on:
1. Brand_Voice_Consistency
2. Contextual_Intelligence
3. Tone_Adaptation_Within_Brand

Return ONLY JSON:
{{
  "scores": {{"Brand_Voice_Consistency": 3, "Contextual_Intelligence": 3, "Tone_Adaptation_Within_Brand": 3}},
  "weighted_score": 3.0,
  "strengths": ["strength"],
  "weaknesses": ["weakness"]
}}"""
        return [
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

    def compare_responses_fully_blind(self, chat_content, baseline_resp, adaptive_resp, context):
        compare_start = time.time()
        responses = [
            {"label": "A", "text": baseline_resp, "actual": "baseline"},
            {"label": "B", "text": adaptive_resp,  "actual": "adaptive"},
        ]
        random.shuffle(responses)
        
        criteria = _EVAL_CFG
        score5_lines = "\n".join(f"   ✓ {l}" for l in criteria["score_5_voice"])
        score1_lines = "\n".join(f"   ✗ {l}" for l in criteria["score_1_voice"])
        system_rules = "\n".join(f"{i+1}. {r}" for i, r in enumerate(criteria["evaluator_system_rules"]))
        
        # Build brand context
        brand_context = f"""
CRITICAL CONTEXT:
{_BRAND_NAME} is NOT a traditional customer service brand. Their ethos is:
{criteria["brand_personality_note"]}

Core Values: {', '.join(_brand['core_values'])}
Messaging Pillar: {_BRAND_PILLAR}

=== EVALUATION CRITERIA FOR {_BRAND_NAME} ===

1. BRAND_VOICE_CONSISTENCY (Is this how {_BRAND_NAME} sounds?)

   SCORE 5 - Perfect {_BRAND_NAME} voice:
{score5_lines}

   SCORE 1 - Generic corporate voice:
{score1_lines}

2. CONTEXTUAL_INTELLIGENCE (Does it understand what {_BRAND_NAME} customers need?)

   SCORE 5: Recognizes customer needs ACTION not explanation. Cuts through quickly. Motivates through challenge.
   SCORE 3: Helpful but too much hand-holding. Over-explains.
   SCORE 1: Lengthy explanations. Excessive caveats. Talks AT customer.

   IMPORTANT: Brevity + insight = HIGH score. Length + detail = LOW score.

3. TONE_ADAPTATION_WITHIN_BRAND (Right {_BRAND_NAME} voice for this specific situation?)

   SCORE 5: Maintains brand voice AND fits the moment. Adjusts intensity appropriately.
   SCORE 3: Maintains brand but misses context, OR adapts but loses brand.
   SCORE 1: Wrong tone for situation AND off-brand.
"""

        prompt = f"""Compare these two responses for {_BRAND_NAME} brand alignment.

{brand_context}

USER MESSAGE: "{chat_content}"
CONTEXT: {context.get("intent")} / {context.get("emotion")}

RESPONSE A: "{responses[0]['text']}"

RESPONSE B: "{responses[1]['text']}"

Evaluate EACH response against {_BRAND_NAME}'s brand (NOT generic customer service standards).
Rate each 1-5 on: Brand_Voice_Consistency, Contextual_Intelligence, Tone_Adaptation_Within_Brand

Calculate total score for each. Declare winner based on TOTAL SCORE.

Return ONLY JSON:
{{
  "response_a_scores": {{"Brand_Voice_Consistency": X, "Contextual_Intelligence": X, "Tone_Adaptation_Within_Brand": X}},
  "response_b_scores": {{"Brand_Voice_Consistency": X, "Contextual_Intelligence": X, "Tone_Adaptation_Within_Brand": X}},
  "response_a_total": X.X,
  "response_b_total": X.X,
  "winner": "Response A" or "Response B",
  "confidence": "high/medium/low",
  "reasoning": "Brief explanation focusing on which response better matches {_BRAND_NAME} brand"
}}"""

        try:
            resp_obj = client.chat.completions.create(
                model=self._cmp_model,
                messages=[
                    {"role": "system", "content": f"""You are a BRAND COMPLIANCE evaluator for {_BRAND_NAME}.

CRITICAL RULES:
{system_rules}

Be STRICT. Evaluate against {_BRAND_NAME} brand standards (punchy, direct, challenging), NOT generic customer service.
Traditional "polite customer service" responses should score LOW on Brand_Voice.
Return ONLY valid JSON."""},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._cmp_temp, 
                max_tokens=self._cmp_max,
                response_format={"type": "json_object"},
            )
            raw = resp_obj.choices[0].message.content
            result = json.loads(self._extract_json(raw))
            
            a_is_baseline = responses[0]["actual"] == "baseline"
            winner_label = result.get("winner", "")
            if winner_label == "Response A":
                actual_winner = "baseline" if a_is_baseline else "adaptive"
            elif winner_label == "Response B":
                actual_winner = "adaptive" if a_is_baseline else "baseline"
            else:
                actual_winner = "tie"
            
            tokens = getattr(resp_obj.usage, "total_tokens", 0) if hasattr(resp_obj, "usage") else 0
            compare_time = time.time() - compare_start
            result["_meta"] = {"model": self._cmp_model, "tokens": tokens, "time": compare_time}
            result["actual_winner"] = actual_winner
            result["baseline_scores"] = result["response_a_scores"] if a_is_baseline else result["response_b_scores"]
            result["adaptive_scores"] = result["response_b_scores"] if a_is_baseline else result["response_a_scores"]
            return result
            
        except Exception as e:
            print(f"Comparison error: {e}")
            compare_time = time.time() - compare_start
            return {
                "actual_winner": "tie", "confidence": "none", "reasoning": "Comparison failed",
                "baseline_scores": {"Brand_Voice_Consistency": 3, "Contextual_Intelligence": 3, "Tone_Adaptation_Within_Brand": 3},
                "adaptive_scores": {"Brand_Voice_Consistency": 3, "Contextual_Intelligence": 3, "Tone_Adaptation_Within_Brand": 3},
                "response_a_total": 9.0, "response_b_total": 9.0,
                "_meta": {"model": "N/A", "tokens": 0, "time": compare_time},
            }


# Initialise evaluator from config
_doc_path = _web.get("brand_doc_path")
evaluator = BrandEvaluator(brand_doc_path=_doc_path if _doc_path else None)


# ── Conversation history ──────────────────────────────────────────────────────
conversation_histories = {}

def get_conversation_history(session_id):
    return [{"user": t["user"], "assistant": t["assistant"]}
            for t in conversation_histories.get(session_id, [])]

def add_to_history(session_id, user_msg, assistant_msg):
    conversation_histories.setdefault(session_id, []).append({
        "user": user_msg, "assistant": assistant_msg,
        "timestamp": datetime.now().isoformat(),
    })


# ── Cost helper ───────────────────────────────────────────────────────────────
_COST_PER_TOKEN = _web["cost_per_token"]

def calculate_cost(model, tokens):
    return tokens * _COST_PER_TOKEN.get(model, 0.000001)


# ── HTML template ─────────────────────────────────────────────────────────────
# Brand name, tagline, pillar, and sample queries all come from config.
def _build_html():
    sample_chips = "\n".join(
        f'                    <div class="sample-chip" onclick="fillQuery(\'{q["message"]}\')">{q["label"]}</div>'
        for q in _web["sample_queries"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_BRAND_NAME} AI Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh; color: #1a202c; padding: 20px;
            background-attachment: fixed;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.98) 100%);
            backdrop-filter: blur(20px); padding: 40px; border-radius: 24px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.3); margin-bottom: 24px;
            border: 2px solid rgba(234,179,8,0.3); border-left: 6px solid #eab308;
        }}
        .header h1 {{
            font-size: 42px; font-weight: 900;
            background: linear-gradient(135deg, #fbbf24 0%, #eab308 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-transform: uppercase; letter-spacing: 1px;
        }}
        .header p {{ font-size: 18px; color: #94a3b8; margin-top: 8px; font-weight: 700; }}
        .tagline {{
            display: inline-block; padding: 8px 20px;
            background: linear-gradient(135deg, #eab308 0%, #fbbf24 100%);
            color: #0f172a; border-radius: 20px; font-size: 14px;
            font-weight: 900; margin-top: 12px; text-transform: uppercase; letter-spacing: 1.5px;
        }}
        .chat-container {{
            background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%);
            backdrop-filter: blur(20px); border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); margin-bottom: 24px;
            display: flex; flex-direction: column;
            height: calc(100vh - 350px); min-height: 500px;
            border: 2px solid rgba(234,179,8,0.2);
        }}
        .chat-header {{
            padding: 24px 28px;
            background: linear-gradient(135deg, #eab308 0%, #fbbf24 100%);
            color: #0f172a; display: flex; justify-content: space-between; align-items: center;
        }}
        .chat-header h2 {{
            font-size: 20px; font-weight: 900; text-transform: capitalize;
            letter-spacing: 1px; display: flex; align-items: center; gap: 10px;
        }}
        .status-indicator {{
            width: 10px; height: 10px; background: #10b981;
            border-radius: 50%; animation: pulse 2s infinite;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            50% {{ box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        .chat-controls {{ display: flex; align-items: center; gap: 20px; }}
        .analysis-toggle {{ display: flex; align-items: center; gap: 12px; font-size: 14px; color: #0f172a; font-weight: 800; }}
        .toggle-switch {{
            position: relative; width: 52px; height: 28px;
            background: rgba(15,23,42,0.3); border-radius: 14px; cursor: pointer;
            transition: all 0.3s; border: 2px solid rgba(15,23,42,0.4);
        }}
        .toggle-switch.active {{ background: #10b981; border-color: #059669; }}
        .toggle-slider {{
            position: absolute; top: 2px; left: 2px;
            width: 20px; height: 20px; background: white;
            border-radius: 50%; transition: transform 0.3s;
        }}
        .toggle-switch.active .toggle-slider {{ transform: translateX(24px); }}
        .btn-clear {{
            padding: 10px 20px; background: rgba(15,23,42,0.4);
            border: 2px solid rgba(15,23,42,0.5); border-radius: 10px;
            font-size: 14px; color: #0f172a; cursor: pointer;
            font-weight: 800; transition: all 0.3s;
        }}
        .btn-clear:hover {{ background: rgba(15,23,42,0.6); }}
        .chat-messages {{
            flex: 1; overflow-y: auto; padding: 28px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        }}
        .message {{ margin-bottom: 20px; display: flex; flex-direction: column; animation: slideIn 0.4s ease; }}
        @keyframes slideIn {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .message.user {{ align-items: flex-end; }}
        .message.assistant {{ align-items: flex-start; }}
        .message-bubble {{
            max-width: 70%; padding: 16px 20px; border-radius: 18px;
            line-height: 1.6; font-size: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        .message.user .message-bubble {{
            background: linear-gradient(135deg, #eab308 0%, #fbbf24 100%);
            color: #0f172a; font-weight: 600; border-bottom-right-radius: 4px;
        }}
        .message.assistant .message-bubble {{
            background: linear-gradient(135deg, #475569 0%, #64748b 100%);
            color: #f8fafc; border-bottom-left-radius: 4px;
        }}
        .message-label {{
            font-size: 11px; color: #94a3b8; margin-bottom: 8px; padding: 0 6px;
            font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;
        }}
        .typing-indicator {{
            display: none; padding: 16px 20px;
            background: rgba(71,85,105,0.8); border-radius: 18px; max-width: 80px;
        }}
        .typing-indicator.active {{ display: flex; gap: 6px; }}
        .typing-dot {{
            width: 10px; height: 10px; border-radius: 50%; background: #eab308;
            animation: typing 1.4s infinite;
        }}
        .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes typing {{
            0%, 60%, 100% {{ opacity: 0.3; transform: translateY(0); }}
            30% {{ opacity: 1; transform: translateY(-6px); }}
        }}
        .chat-input-container {{
            padding: 20px 24px; border-top: 2px solid rgba(234,179,8,0.2);
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        }}
        .input-wrapper {{ display: flex; gap: 16px; align-items: flex-end; }}
        textarea {{
            flex: 1; padding: 16px 20px; border: 2px solid rgba(234,179,8,0.3);
            border-radius: 16px; font-size: 15px; font-family: inherit;
            resize: none; min-height: 50px; max-height: 120px;
            background: rgba(15,23,42,0.6); color: #f8fafc; transition: all 0.3s;
        }}
        textarea:focus {{
            outline: none; border-color: #eab308;
            box-shadow: 0 0 0 4px rgba(234, 179, 8, 0.2); background: rgba(15,23,42,0.8);
        }}
        textarea::placeholder {{ color: #64748b; }}
        .btn-send {{
            background: linear-gradient(135deg, #eab308 0%, #fbbf24 100%);
            color: #0f172a; border: none; padding: 14px 30px;
            border-radius: 14px; font-size: 15px; font-weight: 900;
            cursor: pointer; box-shadow: 0 8px 25px rgba(234, 179, 8, 0.4);
            transition: all 0.3s; text-transform: uppercase;
        }}
        .btn-send:hover {{ transform: translateY(-2px); box-shadow: 0 12px 30px rgba(234, 179, 8, 0.5); }}
        .btn-send:disabled {{ opacity: 0.6; cursor: not-allowed; }}
        .sample-queries {{ display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }}
        .sample-chip {{
            padding: 10px 18px; background: rgba(71,85,105,0.6);
            border: 2px solid rgba(234,179,8,0.3); border-radius: 25px;
            font-size: 13px; cursor: pointer; transition: all 0.3s;
            font-weight: 700; color: #f8fafc;
        }}
        .sample-chip:hover {{
            background: linear-gradient(135deg, #eab308 0%, #fbbf24 100%);
            border-color: #eab308; color: #0f172a; transform: translateY(-2px);
        }}
        .analysis-section {{
            background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%);
            backdrop-filter: blur(20px); border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.2); margin-bottom: 20px;
            overflow: hidden; border: 2px solid rgba(234,179,8,0.2);
        }}
        .section-header {{
            padding: 20px 24px; cursor: pointer;
            background: linear-gradient(135deg, rgba(234,179,8,0.1) 0%, rgba(234,179,8,0.05) 100%);
            display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid rgba(234,179,8,0.1);
        }}
        .section-header h3 {{ font-size: 14px; font-weight: 900; color: #eab308; text-transform: uppercase; letter-spacing: 1px; }}
        .section-toggle {{ color: #eab308; font-size: 18px; transition: transform 0.3s; }}
        .section-content {{ padding: 24px; display: none; }}
        .analysis-section.expanded .section-content {{ display: block; }}
        .analysis-section.expanded .section-toggle {{ transform: rotate(180deg); }}
        .context-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
        .context-card {{
            background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.2);
            border-radius: 12px; padding: 16px;
        }}
        .context-card .label {{ font-size: 11px; color: #94a3b8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }}
        .context-card .value {{ font-size: 16px; font-weight: 800; color: #eab308; }}
        .eval-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .eval-card {{
            background: rgba(15,23,42,0.6); border-radius: 16px; padding: 20px;
            border: 1px solid rgba(234,179,8,0.15);
        }}
        .eval-card h4 {{ font-size: 13px; font-weight: 900; color: #94a3b8; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 1px; }}
        .score-item {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .score-item:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
        .score-name {{ font-size: 13px; font-weight: 700; color: #94a3b8; }}
        .score-value {{ font-size: 20px; font-weight: 900; }}
        .score-value.high {{ color: #10b981; }}
        .score-value.medium {{ color: #eab308; }}
        .score-value.low {{ color: #ef4444; }}
        .total-score {{ text-align: center; padding: 20px; background: rgba(234,179,8,0.08); border-radius: 12px; margin-top: 16px; }}
        .total-score .number {{ font-size: 48px; font-weight: 900; color: #eab308; line-height: 1; }}
        .total-score .label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; font-weight: 800; text-transform: uppercase; }}
        .feedback-list {{ margin-top: 16px; }}
        .feedback-item {{ padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 8px; font-weight: 600; }}
        .feedback-item.strength {{ background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }}
        .feedback-item.weakness {{ background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }}
        .winner-banner {{
            padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;
            text-align: center; font-weight: 900; font-size: 18px;
        }}
        .winner-banner.adaptive {{ background: rgba(16,185,129,0.15); color: #10b981; border: 2px solid rgba(16,185,129,0.3); }}
        .winner-banner.baseline {{ background: rgba(239,68,68,0.15); color: #ef4444; border: 2px solid rgba(239,68,68,0.3); }}
        .winner-banner.tie {{ background: rgba(234,179,8,0.15); color: #eab308; border: 2px solid rgba(234,179,8,0.3); }}
        .perf-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }}
        .perf-stat {{
            background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.2);
            border-radius: 12px; padding: 16px; text-align: center;
        }}
        .perf-stat .perf-value {{ font-size: 28px; font-weight: 900; color: #eab308; }}
        .perf-stat .perf-label {{ font-size: 11px; color: #94a3b8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }}
        .step-breakdown {{ margin-top: 16px; }}
        .step-item {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; padding: 10px 14px; background: rgba(15,23,42,0.5); border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05); }}
        .step-name {{ font-size: 13px; font-weight: 700; color: #94a3b8; }}
        .step-details {{ display: flex; gap: 16px; font-size: 12px; font-weight: 700; color: #64748b; white-space: nowrap; }}
        .step-model {{ padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }}
        .step-model.api {{ background: rgba(234,179,8,0.15); color: #eab308; }}
        .step-model.noapi {{ background: rgba(100,116,139,0.2); color: #64748b; }}
        @media (max-width: 768px) {{
            .eval-grid {{ grid-template-columns: 1fr; }}
            .perf-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{_BRAND_NAME.upper()} ⚡</h1>
            <p>     {_BRAND_PILLAR}</p>
            <span class="tagline">{_BRAND_TAGLINE.upper()}</span>
        </div>

        <div class="chat-container">
            <div class="chat-header">
                <h2>
                    <span class="status-indicator"></span>
                    {_BRAND_NAME} Coach 🏃
                </h2>
                <div class="chat-controls">
                    <div class="analysis-toggle">
                        <span>Show Analysis</span>
                        <div class="toggle-switch active" id="analysisToggle">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <button class="btn-clear" onclick="clearChat()">Back to Zero</button>
                </div>
            </div>

            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div class="message-label">{_BRAND_NAME}</div>
                    <div class="message-bubble">
                        Excuses don't log in. You do. What's up?
                    </div>
                </div>
            </div>

            <div class="chat-input-container">
                <div class="input-wrapper">
                    <textarea id="userInput" placeholder="What's stopping you? Say it..." rows="1"></textarea>
                    <button class="btn-send" onclick="sendMessage()">Go</button>
                </div>
                <div class="sample-queries">
{sample_chips}
                </div>
            </div>
        </div>

        <div id="analysisContainer" style="display: block;">
            <div class="analysis-section expanded">
                <div class="section-header" onclick="toggleSection(this)">
                    <h3>CONTEXT ANALYSIS</h3>
                    <span class="section-toggle">▼</span>
                </div>
                <div class="section-content">
                    <div class="context-grid" id="contextGrid">
                        <div class="context-card"><div class="label">Emotion</div><div class="value" id="ctxEmotion">—</div></div>
                        <div class="context-card"><div class="label">Motivation</div><div class="value" id="ctxMotivation">—</div></div>
                        <div class="context-card"><div class="label">Urgency</div><div class="value" id="ctxUrgency">—</div></div>
                        <div class="context-card"><div class="label">Intent</div><div class="value" id="ctxIntent">—</div></div>
                    </div>
                </div>
            </div>

            <div class="analysis-section" id="evalSection">
                <div class="section-header" onclick="toggleSection(this)">
                    <h3>BRAND EVALUATION</h3>
                    <span class="section-toggle">▼</span>
                </div>
                <div class="section-content">
                    <div class="eval-grid" id="evalGrid">
                        <div class="eval-card">
                            <h4>Baseline (Generic Bot)</h4>
                            <div id="baselineScores"></div>
                            <div class="total-score"><div class="number" id="baselineTotal">—</div><div class="label">Total Score</div></div>
                            <div class="feedback-list" id="baselineFeedback"></div>
                        </div>
                        <div class="eval-card">
                            <h4>{_BRAND_NAME} Adaptive</h4>
                            <div id="adaptiveScores"></div>
                            <div class="total-score"><div class="number" id="adaptiveTotal">—</div><div class="label">Total Score</div></div>
                            <div class="feedback-list" id="adaptiveFeedback"></div>
                        </div>
                    </div>
                    <div id="winnerBanner" style="display:none;" class="winner-banner"></div>
                </div>
            </div>

  

            </div>

                <div class="analysis-section" id="perfSection">
 
                <div class="section-header" onclick="toggleSection(this)">
                    <h3>PERFORMANCE METRICS</h3>
                    <span class="section-toggle">▼</span>
                </div>
                <div class="section-content">
                    <div class="perf-grid">
                        <div class="perf-stat"><div class="perf-value" id="perfTime">—</div><div class="perf-label">Total Time</div></div>
                        <div class="perf-stat"><div class="perf-value" id="perfTokens">—</div><div class="perf-label">Total Tokens</div></div>
                        <div class="perf-stat"><div class="perf-value" id="perfCost">—</div><div class="perf-label">Total Cost</div></div>
                    </div>
                    <div class="step-breakdown" id="stepBreakdown"></div>
                </div>
                </div>
      </div>
    </div>

    

    <script>
        let showAnalysis = true;
        let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        const BRAND_NAME = {json.dumps(_BRAND_NAME)};

        document.getElementById('analysisToggle').addEventListener('click', function() {{
            showAnalysis = !showAnalysis;
            this.classList.toggle('active', showAnalysis);
            document.getElementById('analysisContainer').style.display = showAnalysis ? 'block' : 'none';
document.getElementById('perfSection').style.display = 'block'; // always visible
        }});

        function toggleSection(header) {{
            header.parentElement.classList.toggle('expanded');
        }}

        function fillQuery(text) {{
            document.getElementById('userInput').value = text;
            document.getElementById('userInput').focus();
        }}

        function clearChat() {{
            document.getElementById('chatMessages').innerHTML = `
                <div class="message assistant">
                    <div class="message-label">${{BRAND_NAME}}</div>
                    <div class="message-bubble">Excuses don't log in. You do. What's up?</div>
                </div>`;
            sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            ['ctxEmotion','ctxMotivation','ctxUrgency','ctxIntent','baselineScores','adaptiveScores',
             'baselineFeedback','adaptiveFeedback','stepBreakdown'].forEach(id => {{
                const el = document.getElementById(id);
                if (el) el.innerHTML = '';
            }});
            ['baselineTotal','adaptiveTotal','perfTime','perfTokens','perfCost'].forEach(id => {{
                const el = document.getElementById(id);
                if (el) el.textContent = '—';
            }});
            const wb = document.getElementById('winnerBanner');
            if (wb) wb.style.display = 'none';
        }}

        async function sendMessage() {{
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            addMessage('user', message);
            input.value = '';
            input.style.height = 'auto';
            showTyping();
            document.querySelector('.btn-send').disabled = true;

            try {{
                const res = await fetch('/compare', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ message, session_id: sessionId, show_analysis: showAnalysis }})
                }});
                const data = await res.json();
                hideTyping();
                document.querySelector('.btn-send').disabled = false;

                if (data.is_gratitude) {{
                    addMessage('assistant', data.gratitude_response);
                    return;
                }}

                if (data.error) {{
                    addMessage('assistant', 'Error: ' + data.error);
                    return;
                }}

                if (!data.analysis_mode && !data.adaptive) {{
                    addMessage('assistant', data.adaptive_response || '(no response)');
                    updatePerformance(data.performance);
                    return;
                }}

                addMessage('assistant', data.adaptive.response);
                updateContext(data.adaptive.context);
                updateEvaluation(data.baseline.evaluation, data.adaptive.evaluation, data.comparison, data.baseline.response, data.adaptive.response);
                updatePerformance(data.performance);

            }} catch(err) {{
                hideTyping();
                document.querySelector('.btn-send').disabled = false;
                addMessage('assistant', 'Connection error. Try again.');
            }}
        }}

        function addMessage(role, text) {{
            const container = document.getElementById('chatMessages');
            const label = role === 'user' ? 'You' : BRAND_NAME;
            const div = document.createElement('div');
            div.className = `message ${{role}}`;
            div.innerHTML = `<div class="message-label">${{label}}</div><div class="message-bubble">${{escapeHtml(text)}}</div>`;
            container.appendChild(div);
            scrollToBottom();
        }}

        function showTyping() {{
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = 'message assistant'; div.id = 'typingMsg';
            div.innerHTML = `<div class="message-label">${{BRAND_NAME}}</div>
                <div class="typing-indicator active">
                    <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
                </div>`;
            container.appendChild(div);
            scrollToBottom();
        }}

        function hideTyping() {{
            const t = document.getElementById('typingMsg');
            if (t) t.remove();
        }}

        function updateContext(ctx) {{
            if (!ctx) return;
            document.getElementById('ctxEmotion').textContent    = ctx.emotion || '—';
            document.getElementById('ctxMotivation').textContent = ctx.motivation_level || '—';
            document.getElementById('ctxUrgency').textContent    = ctx.urgency_level || '—';
            document.getElementById('ctxIntent').textContent     = (ctx.intent || '—').replace(/_/g,' ');
            // Auto-expand context section
            const ctxSection = document.querySelector('.analysis-section');
            if (ctxSection && !ctxSection.classList.contains('expanded')) {{
                ctxSection.classList.add('expanded');
            }}
        }}

        function updateEvaluation(baseEval, adaptEval, comparison, baselineText, adaptiveText) {{
            renderScores('baselineScores', baseEval.scores);
            renderScores('adaptiveScores', adaptEval.scores);
            document.getElementById('baselineTotal').textContent = baseEval.weighted_score?.toFixed(1) || '—';
            document.getElementById('adaptiveTotal').textContent = adaptEval.weighted_score?.toFixed(1) || '—';
            renderFeedback('baselineFeedback', baseEval.strengths, baseEval.weaknesses, baselineText, true);
            renderFeedback('adaptiveFeedback', adaptEval.strengths, adaptEval.weaknesses, adaptiveText, false);
            if (comparison) {{
                const wb = document.getElementById('winnerBanner');
                wb.style.display = 'block';
                const w = comparison.actual_winner;
                wb.className = `winner-banner ${{w}}`;
                wb.textContent = w === 'adaptive' ? `✅ ${{BRAND_NAME}} Wins` : w === 'baseline' ? '⚠️ Generic Bot Wins' : '🤝 Tie';
            }}
            document.getElementById('evalSection').classList.add('expanded');
        }}

        function renderScores(containerId, scores) {{
            if (!scores) return;
            const el = document.getElementById(containerId);
            el.innerHTML = Object.entries(scores).map(([k, v]) => {{
                const cls = v >= 4 ? 'high' : v >= 3 ? 'medium' : 'low';
                return `<div class="score-item">
                    <span class="score-name">${{k.replace(/_/g,' ')}}</span>
                    <span class="score-value ${{cls}}">${{v}}/5</span>
                </div>`;
            }}).join('');
        }}

        function renderFeedback(containerId, strengths, weaknesses, fullText, isBaseline) {{
            const el = document.getElementById(containerId);
            let html = '';

            // Full response block with annotated highlights
            if (fullText) {{
                const responseId = containerId + '_resp';
                const bgColor = isBaseline ? 'rgba(239,68,68,0.06)' : 'rgba(16,185,129,0.06)';
                const borderColor = isBaseline ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)';
                const labelColor = isBaseline ? '#ef4444' : '#10b981';
                const label = isBaseline ? '⚠ Full Baseline Response' : '✅ Full Adaptive Response';

                html += `<div style="margin-bottom:14px;">
                    <div style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.8px;color:${{labelColor}};margin-bottom:6px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;"
                         onclick="document.getElementById('${{responseId}}').style.display = document.getElementById('${{responseId}}').style.display==='none'?'block':'none'">
                        ${{label}} <span style="font-size:14px;">▾</span>
                    </div>
                    <div id="${{responseId}}" style="background:${{bgColor}};border:1px solid ${{borderColor}};border-radius:10px;padding:14px;font-size:13px;line-height:1.7;color:#e2e8f0;font-style:italic;white-space:pre-wrap;word-break:break-word;">
                        "${{escapeHtml(fullText)}}"
                    </div>
                </div>`;
            }}

            // Strengths
            if ((strengths||[]).length) {{
                html += (strengths).map(t => `<div class="feedback-item strength">✓ ${{escapeHtml(t)}}</div>`).join('');
            }}

            // Weaknesses / what to change
            if ((weaknesses||[]).length) {{
                if (isBaseline && weaknesses.length) {{
                    html += `<div style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.8px;color:#ef4444;margin:12px 0 6px;">What to Change</div>`;
                }}
                html += (weaknesses).map(t => `<div class="feedback-item weakness">✗ ${{escapeHtml(t)}}</div>`).join('');
            }}

            el.innerHTML = html;
        }}

        function updatePerformance(perf) {{
            if (!perf) return;
            document.getElementById('perfTime').textContent   = perf.total_time?.toFixed(2) + 's';
            document.getElementById('perfTokens').textContent = (perf.total_tokens||0).toLocaleString();
            document.getElementById('perfCost').textContent   = '$' + (perf.total_cost||0).toFixed(6);
            document.getElementById('perfSection').classList.add('expanded');

            if (perf.breakdown) {{
                const container = document.getElementById('stepBreakdown');
                let html = '';
                for (const [stepName, metrics] of Object.entries(perf.breakdown)) {{
                    const model = metrics.model || 'No API call';
                    const modelClass = metrics.model ? 'api' : 'noapi';
                    const t = (metrics.time||0).toFixed(3);
                    const tok = metrics.tokens ? metrics.tokens.toLocaleString() : '—';
                    const cost = metrics.cost ? '$' + metrics.cost.toFixed(6) : '—';
                    html += `<div class="step-item">
                        <div class="step-name">${{escapeHtml(stepName)}}</div>
                        <div class="step-details">
                            <span class="step-model ${{modelClass}}">${{escapeHtml(model)}}</span>
                            <span>${{t}}s</span><span>${{tok}}</span><span>${{cost}}</span>
                        </div>
                    </div>`;
                }}
                container.innerHTML = html;
            }}
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function scrollToBottom() {{
            const c = document.getElementById('chatMessages');
            c.scrollTop = c.scrollHeight;
        }}

        document.getElementById('userInput').addEventListener('keydown', function(e) {{
            if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
        }});
        document.getElementById('userInput').addEventListener('input', function() {{
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        }});
    </script>
</body>
</html>"""


HTML_TEMPLATE = _build_html()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_TEMPLATE


@app.route("/compare", methods=["POST"])
def compare():
    data       = request.json
    message    = data.get("message", "")
    session_id = data.get("session_id", "default")
    show_analysis = data.get("show_analysis", True)

    if not message:
        return jsonify({"error": "No message"}), 400

    try:
        # Gratitude shortcut
        grat_cfg = _web["gratitude"]
        if any(w in message.lower() for w in grat_cfg["keywords"]) and len(message.split()) <= grat_cfg["max_words"]:
            response = random.choice(grat_cfg["responses"])
            add_to_history(session_id, message, response)
            return jsonify({"is_gratitude": True, "gratitude_response": response})

        result           = framework.process_message(message=message, session_id=session_id)
        adaptive_response = result.response_text
        context_analysis  = result.context_analysis

        performance_data = {
            "total_time":   result.total_time,
            "total_tokens": result.total_tokens,
            "total_cost":   result.total_cost,
            "breakdown":    result.breakdown,
        }

        history = get_conversation_history(session_id)

        add_to_history(session_id, message, adaptive_response)

        if not show_analysis:
            return jsonify({"analysis_mode": False, "adaptive_response": adaptive_response, "performance": performance_data})

        # Full analysis
        baseline_result = baseline_system.generate_response(message, conversation_history=history)
        context_for_eval = {
            "intent":  context_analysis.get("intent", "question"),
            "emotion": str(context_analysis.get("emotion", "neutral")).capitalize(),
        }

        baseline_eval = evaluator.evaluate_response_blind(message, baseline_result["response"], context_for_eval)
        adaptive_eval = evaluator.evaluate_response_blind(message, adaptive_response, context_for_eval)
        comparison    = evaluator.compare_responses_fully_blind(message, baseline_result["response"], adaptive_response, context_for_eval)

        performance_data["breakdown"]["Baseline Response"]     = {"time": baseline_result["time"],    "model": baseline_result["model"], "tokens": baseline_result["tokens"], "cost": calculate_cost(baseline_result["model"], baseline_result["tokens"])}
        performance_data["breakdown"]["Baseline Evaluation"]   = baseline_eval["_meta"]
        performance_data["breakdown"]["Adaptive Evaluation"]   = adaptive_eval["_meta"]
        performance_data["breakdown"]["Comparison Evaluation"] = comparison["_meta"]

        performance_data["total_time"] += (baseline_result["time"] + baseline_eval["_meta"].get("time", 0) + adaptive_eval["_meta"].get("time", 0) + comparison["_meta"].get("time", 0))
        performance_data["total_tokens"] += (baseline_result["tokens"] + baseline_eval["_meta"].get("tokens", 0) + adaptive_eval["_meta"].get("tokens", 0) + comparison["_meta"].get("tokens", 0))
        performance_data["total_cost"] += sum(calculate_cost(x["_meta"].get("model","N/A"), x["_meta"].get("tokens",0)) for x in [baseline_eval, adaptive_eval, comparison]) + calculate_cost(baseline_result["model"], baseline_result["tokens"])

        return jsonify({
            "analysis_mode": True,
            "baseline": {"response": baseline_result["response"], "evaluation": baseline_eval},
            "adaptive": {
                "response": adaptive_response,
                "context": {
                    "emotion":          context_analysis.get("emotion", "neutral").capitalize(),
                    "motivation_level": str(context_analysis.get("motivation_level", "medium")).capitalize(),
                    "urgency_level":    str(context_analysis.get("urgency_level", "normal")).capitalize(),
                    "intent":           str(context_analysis.get("intent", "question")).replace("_", " ").title(),
                },
                "evaluation": adaptive_eval,
            },
            "comparison":  comparison,
            "performance": performance_data,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(f"{_BRAND_NAME} - ADAPTIVE WEB INTERFACE")
    print("=" * 70)
    print(f"\n Open: http://localhost:{_web['port']}")
    print("=" * 70 + "\n")
    app.run(debug=_web["debug"], port=_web["port"])