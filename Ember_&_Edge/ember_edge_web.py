"""
Ember & Edge Web Interface - 10/10 VERSION
Matches fixed config with brand-perfect responses
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from flask import Flask, request, jsonify
import json
import secrets
from datetime import datetime
import time
import re
import os
from openai import OpenAI
from ember_edge_framework import UniversalAdaptiveFramework
from config_loader import build_brand_ethos
from context_understanding_engine import ContextUnderstandingEngine
from config_loader import build_brand_ethos, cfg

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
EMBER_EDGE_ETHOS = build_brand_ethos()
framework = UniversalAdaptiveFramework(client)
class BaselineSystem:
    def __init__(self, brand_ethos):
        self.client = client
        self.temperature = 0.7
        self.max_tokens = 350
        self._web_cfg = cfg("web")

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
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            'response': response.choices[0].message.content,
            'model': "gpt-4o-mini",
            'tokens': response.usage.total_tokens,
            'time': time.time() - step_start
        }

baseline_system = BaselineSystem(EMBER_EDGE_ETHOS)

class EmberEdgeBrandEvaluator:
    """Evaluates responses against brand standards - 3 CORE PARAMETERS"""
    def __init__(self, client):
        self.client = client
        self.EVALUATION_PARAMETERS = {
            "Brand_Authenticity": {"weight": 1.0},
            "Context_Awareness": {"weight": 1.0},
            "Tone_Adaptation": {"weight": 1.0}
        }
    
    def _extract_json(self, text):
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    
    def evaluate_response_blind(self, chat_content, response, context):
        eval_start = time.time()
        
        prompt = f"""Evaluate this Ember & Edge knife brand response on THREE core dimensions (1-5 scale):

MESSAGE: "{chat_content}"
RESPONSE: "{response}"
CONTEXT: User emotion = {context.get('emotion', 'neutral')}, Intent = {context.get('intent', 'question')}, Cook type = {context.get('cook_type', 'home_cook')}

Rate ONLY these 3 parameters:

1. **Brand_Authenticity** (1-5):
- Must feel like a patient chef-instructor with hands-on experience (not corporate or generic customer service language).
- Uses sensory, tactile, and experiential language (feel, whisper, glide, balance).
- Avoids generic chatbot phrases like "Does that make sense?", "Let me know if...", "Feel free to..."
- NO technical specs (HRC, geometry) - redirects to experience instead
- Flows naturally and engages user as a craft mentor, not as a manual.
- Higher scores for responses that evoke brand persona; lower scores for generic or formulaic responses.

2. **Context_Awareness** (1-5):
- Does the response feel attuned to the user's current situation or mindset?
- Does it align tone with Ember & Edge brand persona appropriately?
- For complaints: Is it solution-focused without poetry?
- For professionals: Is it direct without specs?
- For beginners: Is it reassuring without condescension?
- Higher score for subtle, human, brand-aligned awareness; lower for mechanical responses.

3. **Tone_Adaptation** (1-5):
- Poetry level matches context (NO poetry for complaints/escalations)?
- Directness appropriate (high for complaints, balanced for inquiries)?
- Questions used appropriately (diagnostic for complaints, reflective for guidance)?
- Sensory language fits the situation (YES for product questions, NO for urgent issues)?
- Professional chefs get direct performance talk, not sensory reflections?

Return ONLY this JSON structure:
{{
  "scores": {{
    "Brand_Authenticity": 3,
    "Context_Awareness": 3,
    "Tone_Adaptation": 3
  }},
  "weighted_score": 3.0,
  "strengths": ["Specific strength 1", "Specific strength 2"],
  "weaknesses": ["Specific weakness 1", "Specific weakness 2"]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "You are an expert brand evaluator. Return ONLY valid JSON with no preamble."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_completion_tokens=600
            )
            
            result = json.loads(self._extract_json(resp.choices[0].message.content))
            
            if "weighted_score" not in result:
                weights = self.EVALUATION_PARAMETERS
                total = sum(result["scores"].get(k, 3) * weights[k]["weight"] for k in weights)
                result["weighted_score"] = round(total / sum(w["weight"] for w in weights.values()), 2)
            
            result['_meta'] = {
                'model': "gpt-5.1",
                'tokens': resp.usage.total_tokens,
                'time': time.time() - eval_start
            }
            return result
            
        except Exception as e:
            return {
                "scores": {k: 3 for k in self.EVALUATION_PARAMETERS},
                "weighted_score": 3.0,
                "strengths": ["Error"],
                "weaknesses": [str(e)],
                "_meta": {"model": "gpt-5.1", "tokens": 0, "time": time.time() - eval_start}
            }
    
    def compare_responses_fully_blind(self, chat_content, baseline_resp, adaptive_resp, context):
        import random
        
        responses = [
            {"label": "A", "text": baseline_resp, "actual": "baseline"},
            {"label": "B", "text": adaptive_resp, "actual": "adaptive"}
        ]
        random.shuffle(responses)
        
        # BUILD COMPREHENSIVE BRAND CONTEXT (matching evaluate_response_blind detail level)
        brand_context = """
=== EMBER & EDGE BRAND IDENTITY ===

**WHO WE ARE**: Patient chef-instructor with 20 years in Michelin kitchens. Teaching without lecturing, poetic without pretentious, calm without cold, knowledgeable without condescending.

**CORE PHILOSOPHY**:
- "The Knife Serves the Ingredient" - Tool is in service of the food, not the cook's ego
- "Feel Before Technique" - Sensory experience precedes technical knowledge
- "Every Cut Tells a Story" - Each cooking moment is meaningful
- "Forged, Not Made" - Craftsmanship over manufacturing

**VOICE CHARACTERISTICS**:
✓ Sensory, tactile language (feel, whisper, glide, balance)
✓ Natural flowing sentences (8-15 words average)
✓ Teaching through experience and contrast
✓ Poetic when appropriate (product questions, not complaints)
✓ Decisive recommendations - name specific knives
✓ Ingredient-focused: "What are you cutting?" before "Which knife?"

✗ NO technical specs (HRC, angles, geometry) as selling points
✗ NO generic chatbot phrases ("Does that make sense?", "Let me know if...", "Feel free to...")
✗ NO corporate/robotic language
✗ NO poetry or sensory language when handling complaints/urgent issues
✗ NO asking "What do you feel?" when user is complaining

**CONTEXT-SPECIFIC RULES**:
- Complaints: Direct, solution-focused, NO poetry
- Professionals: Direct performance talk, NO sensory reflections
- Beginners: Reassuring without condescension
- Product questions: Sensory language encouraged
"""
        
        prompt = f"""Compare these two responses for Ember & Edge knife brand.

{brand_context}

MESSAGE: "{chat_content}"
CONTEXT: User emotion = {context.get('emotion', 'neutral')}, Intent = {context.get('intent', 'question')}, Cook type = {context.get('cook_type', 'home_cook')}

RESPONSE A: "{responses[0]['text']}"
RESPONSE B: "{responses[1]['text']}"

Evaluate BOTH responses against Ember & Edge brand standards (NOT generic customer service).

Rate each on THREE dimensions (1-5):

1. **Brand_Authenticity**: 
   SCORE 5: Patient chef-instructor voice, sensory language when appropriate, flows naturally (8-15 word sentences), NO generic chatbot phrases
   SCORE 3: Somewhat brand-aligned but generic in places
   SCORE 1: Corporate customer service, robotic, formulaic

2. **Context_Awareness**:
   SCORE 5: Perfectly attuned to user's situation - poetry level, directness, and focus all appropriate for context
   SCORE 3: Generally appropriate but misses some context cues
   SCORE 1: Tone-deaf or mechanical, doesn't adapt to situation

3. **Tone_Adaptation**:
   SCORE 5: Poetry/sensory language matches context (YES for product questions, NO for complaints), directness appropriate, questions fit situation
   SCORE 3: Mostly appropriate but some mismatch
   SCORE 1: Wrong tone for situation (e.g., poetic during complaint, or too corporate for product question)

IMPORTANT: 
- Generic polite customer service = LOW scores (1-2) on Brand_Authenticity
- Sensory language in complaints = LOW scores on Tone_Adaptation
- Chef-instructor voice with natural flow = HIGH scores

Calculate total for each. Declare winner based on TOTAL SCORE and brand alignment.

Return JSON:
{{
  "response_a_scores": {{"Brand_Authenticity": X, "Context_Awareness": X, "Tone_Adaptation": X}},
  "response_a_total": X,
  "response_b_scores": {{"Brand_Authenticity": X, "Context_Awareness": X, "Tone_Adaptation": X}},
  "response_b_total": X,
  "winner": "Response A" or "Response B" or "Tie",
  "reasoning": "Which response better embodies Ember & Edge brand",
  "key_differences": ["Difference 1", "Difference 2"]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a BRAND COMPLIANCE evaluator for Ember & Edge.

CRITICAL: Evaluate against Ember & Edge standards (patient chef-instructor, sensory language, craft focus), NOT generic customer service standards.

Traditional polite customer service = LOW Brand_Authenticity scores.
Sensory language in complaints = LOW Tone_Adaptation scores.
Natural chef-instructor voice = HIGH scores.

Return ONLY valid JSON."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result = json.loads(self._extract_json(resp.choices[0].message.content))
            
            a_is_baseline = responses[0]['actual'] == 'baseline'
            winner_label = result.get('winner', '')
            
            if winner_label == 'Response A':
                actual_winner = "baseline" if a_is_baseline else "adaptive"
            elif winner_label == 'Response B':
                actual_winner = "adaptive" if a_is_baseline else "baseline"
            else:
                actual_winner = "tie"
            
            result['baseline_scores'] = result.get('response_a_scores' if a_is_baseline else 'response_b_scores', {})
            result['adaptive_scores'] = result.get('response_b_scores' if a_is_baseline else 'response_a_scores', {})
            result['actual_winner'] = actual_winner
            
            baseline_total = result.get('response_a_total' if a_is_baseline else 'response_b_total', 9)
            adaptive_total = result.get('response_b_total' if a_is_baseline else 'response_a_total', 9)
            
            if baseline_total > 0:
                improvement = ((adaptive_total - baseline_total) / baseline_total) * 100
                result['improvement_percentage'] = f"{improvement:+.1f}%"
            
            result['_meta'] = {'model': "gpt-4o-mini", 'tokens': resp.usage.total_tokens, 'time': 0}
            return result
            
        except Exception as e:
            default_scores = {k: 3 for k in self.EVALUATION_PARAMETERS}
            return {
                "actual_winner": "tie",
                "improvement_percentage": "N/A",
                "reasoning": f"Error: {str(e)}",
                "baseline_scores": default_scores,
                "adaptive_scores": default_scores,
                "_meta": {"model": "N/A", "tokens": 0, "time": 0}
            }

evaluator = EmberEdgeBrandEvaluator(client)
conversation_histories = {}

def get_conversation_history(session_id):
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return [{'user': t['user'], 'assistant': t['assistant']} for t in conversation_histories[session_id]]

def add_to_history(session_id, user_message, assistant_message):
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    conversation_histories[session_id].append({
        'user': user_message,
        'assistant': assistant_message,
        'timestamp': datetime.now().isoformat()
    })
    if len(conversation_histories[session_id]) > 10:
        conversation_histories[session_id] = conversation_histories[session_id][-10:]

@app.route('/')
def index():
    template_path = os.path.join(SCRIPT_DIR, 'ember_edge_template.html')
    with open(template_path, encoding='utf-8') as f:
        return f.read()

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    show_analysis = data.get('show_analysis', True)
    
    if not message:
        return jsonify({'error': 'No message'}), 400
    
    start_time = time.time()
    all_steps = []
    
    try:
        history = get_conversation_history(session_id)
        
        # Check gratitude
        if any(word in message.lower() for word in ['thank', 'thanks', 'perfect', 'yup']) and len(message.split()) <= 5:
            responses = [
                "The kitchen is waiting.",
                "Back to the cutting board.",
                "Feel the blade. Trust your hands.",
                "Quiet hands. Sharp edge."
            ]
            import random
            response = random.choice(responses)
            add_to_history(session_id, message, response)
            
            return jsonify({
                'is_gratitude': True,
                'gratitude_response': response,
                'performance': {
                    'total_time': time.time() - start_time,
                    'steps': []
                }
            })
        
        # Process adaptive response
        print("\n🔪 Processing Ember & Edge adaptive response...")
        
        adaptive_full_start = time.time()
        
        adaptive_result = framework.process_message(message=message, session_id=session_id)
        total_adaptive_time = time.time() - adaptive_full_start
        
        adaptive_response = adaptive_result.response_text
        context_analysis = adaptive_result.context_analysis
        adaptive_tokens = adaptive_result.total_tokens
        adaptive_gen_time = adaptive_result.total_time
        
        add_to_history(session_id, message, adaptive_response)
        
        all_steps.append({
            'name': 'Context Analysis',
            'model': 'Rule-based',
            'tokens': 0,
            'time': 0.001
        })
        
        all_steps.append({
            'name': 'Tone Selection',
            'model': 'Rule-based',
            'tokens': 0,
            'time': 0.001
        })
        
        all_steps.append({
            'name': 'Adaptive Response Generation',
            'model': 'gpt-4o-mini',
            'tokens': adaptive_tokens,
            'time': adaptive_gen_time
        })
        
        if not show_analysis:
            total_time = time.time() - start_time
            total_api_calls = 1
            
            return jsonify({
                'analysis_mode': False,
                'adaptive_response': adaptive_response,
                'performance': {
                    'total_time': total_time,
                    'total_tokens': adaptive_tokens,
                    'total_api_calls': total_api_calls,
                    'steps': all_steps
                }
            })
        
        # Full analysis mode
        baseline_result = baseline_system.generate_response(message, history)
        all_steps.append({
            'name': 'Baseline Generation',
            'model': baseline_result['model'],
            'tokens': baseline_result['tokens'],
            'time': baseline_result['time']
        })
        
        context_for_eval = {
            'intent': context_analysis.get('intent', 'question'),
            'emotion': context_analysis.get('emotion', 'neutral'),
            'cook_type': context_analysis.get('cook_type', 'home_cook'),
        }
        
        baseline_eval = evaluator.evaluate_response_blind(message, baseline_result['response'], context_for_eval)
        if '_meta' in baseline_eval:
            all_steps.append({
                'name': 'Baseline Evaluation',
                'model': baseline_eval['_meta']['model'],
                'tokens': baseline_eval['_meta']['tokens'],
                'time': baseline_eval['_meta']['time']
            })
        
        adaptive_eval = evaluator.evaluate_response_blind(message, adaptive_response, context_for_eval)
        if '_meta' in adaptive_eval:
            all_steps.append({
                'name': 'Adaptive Evaluation',
                'model': adaptive_eval['_meta']['model'],
                'tokens': adaptive_eval['_meta']['tokens'],
                'time': adaptive_eval['_meta']['time']
            })
        
        comparison = evaluator.compare_responses_fully_blind(message, baseline_result['response'], adaptive_response, context_for_eval)
        if '_meta' in comparison:
            all_steps.append({
                'name': 'Response Comparison',
                'model': comparison['_meta']['model'],
                'tokens': comparison['_meta']['tokens'],
                'time': comparison['_meta']['time']
            })
        
        # Calculate totals
        total_time = sum(step['time'] for step in all_steps)
        total_tokens = sum(step['tokens'] for step in all_steps)
        total_api_calls = sum(1 for step in all_steps if step['tokens'] > 0)
        
        context_data = {
            'emotion': context_analysis.get('emotion', 'neutral').capitalize(),
            'intent': str(context_analysis.get('intent', 'question')).replace('_', ' ').title(),
            'cook_type': context_analysis.get('cook_type', 'home_cook').replace('_', ' ').title(),
            'knife_knowledge': context_analysis.get('technical_level', 'intermediate').capitalize(),
            'mentioned_knives': context_analysis.get('mentioned_knives', []),
            'mentioned_ingredients': context_analysis.get('mentioned_ingredients', [])
        }
        
        return jsonify({
            'out_of_scope': False,
            'is_gratitude': False,
            'baseline': {
                'response': baseline_result['response'],
                'evaluation': baseline_eval
            },
            'adaptive': {
                'response': adaptive_response,
                'context': context_data,
                'evaluation': adaptive_eval
            },
            'comparison': comparison,
            'performance': {
                'total_time': total_time,
                'total_tokens': total_tokens,
                'total_api_calls': total_api_calls,
                'steps': all_steps
            }
        })
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" Ember & Edge")
    print("="*70 + "\n")
    print("\nOpen: http://localhost:5002")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5002)