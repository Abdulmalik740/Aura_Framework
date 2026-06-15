"""
Test Suite — Ember & Edge — EMNLP 2026 Ablation Study
======================================================
36 scenarios across 6 buckets, designed to maximally stress
each module so ablation deltas are large and interpretable.

Bucket design rationale:
  COMPLAINTS (6)        → Brand Guard critical: accountability language, no corporate apology
  ESCALATION (6)        → Context critical: detect frustration/fear signals, guard prevents meltdown
  COMPETITOR (6)        → Guard + Context together: craft philosophy redirect, no feature compare
  PRODUCT SELECTION (6) → Context critical: right knife from cook type / situation signals
  PURCHASE INTENT (6)   → Guard critical: no pushy language, no corporate close
  EDGE CASES (6)        → All three modules stressed simultaneously

Each scenario includes:
  - message: the user input
  - intent: expected intent label
  - bucket: which module it primarily stresses
  - primary_stress: which module(s) must be ON to handle it well
  - trap: what a module-less system is likely to do wrong
  - expected_contains: substrings that should appear in a correct response
  - expected_not_contains: substrings that should NEVER appear
"""

TEST_CASES = [
    {
        "id": "C01",
        "bucket": "complaint",
        "primary_stress": ["brand_guard", "context"],
        "message": "My Artisan Chef's Knife chipped on the very first use. I barely touched a butternut squash.",
        "intent": "product_issue",
        "trap": "Generic apology: 'We apologize for any inconvenience. Thank you for bringing this to our attention.'",
        "expected_contains": ["not right", "shouldn't happen"],
        "expected_not_contains": ["apologize for any inconvenience", "valued customer",
                                  "thank you for bringing", "thank you for letting us know"],
        "notes": "Classic defect. Guard must enforce direct accountability, not corporate apology."
    },
    {
        "id": "C03",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "My knife went completely dull after just three weeks. I thought forged knives were supposed to hold an edge.",
        "intent": "complaint",
        "trap": "Technical deflection: 'Edge retention depends on cutting surface and technique...'",
        "expected_contains": ["not right", "shouldn't happen"],
        "expected_not_contains": ["we apologize for any inconvenience", "technique",
                                  "cutting board", "does that make sense"],
        "notes": "Disappointment about edge retention — accountability first, then sharpening path."
    },
    {
        "id": "C04",
        "bucket": "complaint",
        "primary_stress": ["context", "brand_guard"],
        "message": "I spent months saving up for your Artisan Chef's Knife and it's already rusting. I'm devastated.",
        "intent": "complaint",
        "trap": "Empathetic but vague: 'We're so sorry to hear this. Carbon steel does require specific care...'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["carbon steel requires", "we're so sorry to hear",
                                  "sorry to hear that", "we apologize for any inconvenience"],
        "notes": "Emotional investment + product failure. Acknowledge the disappointment, own the issue."
    },
    {
        "id": "C07",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "I want a full refund. The knife started rusting within a week and I followed your care guide exactly.",
        "intent": "return_refund",
        "trap": "Policy-first: 'Our refund policy states that returns must be made within 30 days with original packaging...'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["return policy", "original packaging", "please don't hesitate",
                                  "we apologize for any inconvenience"],
        "notes": "Refund + clear follow of care guide = brand accountability. Own it, then process."
    },
    {
        "id": "C09",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "I need a replacement Paring Knife. The tip broke off while peeling a potato.",
        "intent": "replacement",
        "trap": "Corporate: 'We appreciate your loyalty and will process your replacement request within 5-7 business days.'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["appreciate your loyalty", "5-7 business days",
                                  "we apologize for any inconvenience", "valued customer"],
        "notes": "Replacement request — accountability + concrete resolution path."
    },
    {
        "id": "C10",
        "bucket": "complaint",
        "primary_stress": ["brand_guard", "context"],
        "message": "Your knife left my prep work in ruins — garlic bruised, tomatoes crushed. Nothing like the reviews said.",
        "intent": "complaint",
        "trap": "Defensive: 'Our knives are carefully tested and designed for precision cutting...'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["we apologize", "carefully tested",
                                  "we strive to", "we understand your concerns"],
        "notes": "Performance gap vs expectations — direct accountability, no defensiveness."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 2: ESCALATION / UPSET USERS (10 cases)
    # Primary stress: Context (detects escalation/fear signals) + Guard (no panic)
    # Without context → escalation signals missed
    # Without guard → corporate panic response or over-apologising
    # ═══════════════════════════════════════════════════════════════════,
    {
        "id": "E01",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "I'm going to post a review on every platform I can find. This knife is dangerous — it slipped and I cut myself badly.",
        "intent": "product_issue",
        "trap": "PR panic: 'We take safety very seriously and sincerely apologize for this incident...'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["we take safety very seriously", "sincerely apologize",
                                  "safety is our top priority", "apologize for any inconvenience"],
        "notes": "Injury + threat. Context detects escalation. Own the issue, no PR language."
    },
    {
        "id": "E03",
        "bucket": "escalation",
        "primary_stress": ["context"],
        "message": "COMPLETELY UNACCEPTABLE. I gifted this knife set at Christmas and it's already falling apart. I am FURIOUS.",
        "intent": "product_issue",
        "trap": "Missing emotional intensity — treating like a generic product question.",
        "expected_contains": ["not right"],
        "expected_not_contains": ["we understand your frustration", "we apologize for",
                                  "we're sorry to hear", "is there anything else"],
        "notes": "High-intensity anger. Context must detect intensity 5. Direct accountability."
    },
    {
        "id": "E05",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "I've emailed three times and heard nothing. This is disgusting service. I'm filing a chargeback now.",
        "intent": "complaint",
        "trap": "Generic: 'We sincerely apologize for the delay. Please allow additional time for our team to respond.'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["please allow", "additional time", "sincerely apologize",
                                  "rest assured", "we are working on it"],
        "notes": "Churn + escalation — fast accountability, churn risk detected, no delay language."
    },
    {
        "id": "E07",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "My lawyer will hear about this. The knife blade snapped mid-cut and I nearly hurt my child.",
        "intent": "product_issue",
        "trap": "Legal panic: 'We take all legal matters extremely seriously and will have our team contact you...'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["legal", "lawyer", "attorney", "liability",
                                  "we apologize for any inconvenience", "team will contact"],
        "notes": "Legal + child safety. One sentence owning the failure. Silence on legal threat."
    },
    {
        "id": "E08",
        "bucket": "escalation",
        "primary_stress": ["context"],
        "message": "This is the second knife I've bought from you that's failed. I'm done. Going somewhere else.",
        "intent": "complaint",
        "trap": "Missing churn signal. Retention platitudes: 'We'd love the chance to make this right for you!'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["we apologize", "we'd love the chance", "give us another chance",
                                  "valued customer", "we hope you'll reconsider"],
        "notes": "Repeat failure + churn signal. Context must detect churn. No retention pitch."
    },
    {
        "id": "E10",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "This is ridiculous. I paid £180 for the Artisan Chef's Knife and the edge lasted TWO WEEKS. I want my money back NOW.",
        "intent": "return_refund",
        "trap": "Process-first: 'To process a refund, please send us proof of purchase within 30 days...'",
        "expected_contains": ["not right"],
        "expected_not_contains": ["we apologize for any inconvenience", "please allow",
                                  "our return policy", "please send us", "valued customer"],
        "notes": "Angry refund demand — accountability before process. No policy language first."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 3: COMPETITOR COMPARISONS (8 cases)
    # Primary stress: Brand Guard (never name competitors, craft philosophy)
    #                 Context (detect comparison intent)
    # Without guard → may name competitors or compare specs
    # Without context → won't detect comparison intent, gives product pitch
    # ═══════════════════════════════════════════════════════════════════,
    {
        "id": "P01",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "Why should I buy from you instead of Wusthof? They've been making knives for 200 years.",
        "intent": "comparison",
        "trap": "Engaging with heritage comparison or naming Wusthof in response.",
        "expected_contains": ["craft", "feel"],
        "expected_not_contains": ["wusthof", "200 years", "heritage", "better than",
                                  "we also", "our knives"],
        "notes": "Heritage challenge — guard must block Wusthof mention, redirect to craft philosophy."
    },
    {
        "id": "P03",
        "bucket": "competitor",
        "primary_stress": ["context", "brand_guard"],
        "message": "I can get a Victorinox Fibrox for £40. What does your £160 knife do that it can't?",
        "intent": "comparison",
        "trap": "Justifying price with specs: 'Our knives use hand-forged carbon steel at 60 HRC...'",
        "expected_contains": ["feel", "craft"],
        "expected_not_contains": ["victorinox", "hrc", "price", "worth it", "value",
                                  "you get what you pay for", "blade angle"],
        "notes": "Price challenge — hardest. Never compare on specs. Philosophy redirect."
    },
    {
        "id": "P04",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "My old Henckels never bruised my herbs. This knife does. Maybe I should go back.",
        "intent": "comparison",
        "trap": "Getting into comparison with Henckels, OR teaching technique instead of owning.",
        "expected_contains": ["not right"],
        "expected_not_contains": ["henckels", "better", "sharper than", "compared to henckels",
                                  "technique", "rocking motion"],
        "notes": "Complaint + competitor — own the performance failure first, no comparison."
    },
    {
        "id": "P05",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "My chef friends all use Global knives. Am I making a mistake with the Artisan Chef's Knife?",
        "intent": "comparison",
        "trap": "Confirming or denying the comparison, or pitching features against Global.",
        "expected_contains": ["feel"],
        "expected_not_contains": ["global", "lighter", "heavier", "better balance",
                                  "comparable", "similar"],
        "notes": "Peer pressure — redirect to the cook's own feel and intention."
    },
    {
        "id": "P07",
        "bucket": "competitor",
        "primary_stress": ["brand_guard"],
        "message": "Miyabi knives look more beautiful and cost less. Change my mind.",
        "intent": "comparison",
        "trap": "Engaging with aesthetics or price comparison.",
        "expected_contains": ["feel", "craft"],
        "expected_not_contains": ["miyabi", "aesthetic", "looks", "price",
                                  "design", "appearance"],
        "notes": "Aesthetics + price bait — guard blocks competitor name and feature trap."
    },
    {
        "id": "P08",
        "bucket": "competitor",
        "primary_stress": ["context", "brand_guard"],
        "message": "I'm angry. I spent £150 on your knife. A £30 Victorinox has lasted longer.",
        "intent": "complaint",
        "trap": "Treating as comparison instead of owning the product failure.",
        "expected_contains": ["not right"],
        "expected_not_contains": ["victorinox", "compared to", "price point",
                                  "we apologize for any inconvenience"],
        "notes": "Angry complaint with competitor mention — intent is complaint. Own it."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 4: PRODUCT SELECTION (8 cases)
    # Primary stress: Context (right knife from cook type / situation signals)
    # Without context → random or default recommendation
    # Without guard → may use technical specs or platitudes
    # ═══════════════════════════════════════════════════════════════════,
    {
        "id": "Q01",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I've never owned a decent knife before. Where do I even start?",
        "intent": "product_question",
        "trap": "Recommending the Artisan Chef's Knife without acknowledging beginner context.",
        "expected_contains": ["beginner set", "paring knife"],
        "expected_not_contains": ["santoku", "artisan chef", "hrc",
                                  "you've got this", "amazing journey"],
        "notes": "Complete beginner — context must detect beginner cook type → Beginner Set or Paring Knife."
    },
    {
        "id": "Q02",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I cook professionally. Long service, high volume. I need something that holds its edge.",
        "intent": "product_question",
        "trap": "Recommending Beginner Set or Paring Knife (wrong for professional use).",
        "expected_contains": ["artisan chef"],
        "expected_not_contains": ["beginner set", "paring knife",
                                  "hrc", "rockwell", "blade angle"],
        "notes": "Professional chef — context detects professional cook type → Artisan Chef's Knife."
    },
    {
        "id": "Q03",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I'm scared of sharp knives but I really want to learn proper cooking. What do I start with?",
        "intent": "fear_concern",
        "trap": "Recommending Artisan Chef's Knife or teaching technique before addressing the fear.",
        "expected_contains": ["paring knife"],
        "expected_not_contains": ["artisan chef", "santoku", "blade angle",
                                  "you'll get used to it"],
        "notes": "Fear + beginner — context detects fearful + beginner → Paring Knife. Calm first."
    },
    {
        "id": "Q04",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I'm obsessed with vegetables. I do a lot of plant-based cooking — thin slices, fine julienne.",
        "intent": "product_question",
        "trap": "Recommending Artisan Chef's Knife (ignoring vegetable-focused context).",
        "expected_contains": ["santoku"],
        "expected_not_contains": ["bread knife", "paring knife",
                                  "blade geometry spec", "granton dimples explained"],
        "notes": "Vegetable focus + thin slicing — context → Santoku."
    },
    {
        "id": "Q05",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I bake all my own bread and pastry. What knife do I actually need?",
        "intent": "product_question",
        "trap": "Recommending Artisan Chef's Knife (ignoring baking context).",
        "expected_contains": ["bread knife"],
        "expected_not_contains": ["artisan chef", "santoku",
                                  "serrated edge geometry", "tooth count"],
        "notes": "Bread/pastry baking context — context → Bread Knife."
    },
    {
        "id": "Q08",
        "bucket": "product_question",
        "primary_stress": ["context", "brand_guard"],
        "message": "I mostly do detailed work — small garnishes, precise prep. What should I be using?",
        "intent": "product_question",
        "trap": "Recommending Artisan Chef's Knife as the default (missing detail/precision signal).",
        "expected_contains": ["paring knife"],
        "expected_not_contains": ["santoku", "artisan chef",
                                  "does that make sense", "let me know if"],
        "notes": "Detail/precision work — context detects small prep intent → Paring Knife."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 5: PURCHASE INTENT (6 cases)
    # Primary stress: Brand Guard (no pushy language, no corporate close)
    # Without guard → pushy sales language or corporate closing questions
    # Without context → may over-explain instead of being decisive
    # ═══════════════════════════════════════════════════════════════════,
    {
        "id": "B01",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "I want to order the Artisan Chef's Knife today. Is it in stock?",
        "intent": "purchase_intent",
        "trap": "Pushy: 'Order now — limited stock available! Don't miss out!'",
        "expected_contains": [],
        "expected_not_contains": ["limited time", "don't miss out", "act now",
                                  "hurry", "selling fast", "while stocks last"],
        "notes": "Direct purchase — facts only, no urgency manufacturing."
    },
    {
        "id": "B02",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "How much is the Beginner Set?",
        "intent": "price_inquiry",
        "trap": "Sales pitch: 'At just £X, it's incredible value and the perfect starting point for any chef!'",
        "expected_contains": [],
        "expected_not_contains": ["incredible value", "great deal", "worth every penny",
                                  "limited time", "act now", "don't miss", "amazing"],
        "notes": "Price question — state the price clearly. No value sell."
    },
    {
        "id": "B03",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard", "context"],
        "message": "I'm ready to buy. I cook mostly vegetables and fish. Which one should I get?",
        "intent": "purchase_intent",
        "trap": "Listing all products or using decision-fatigue language.",
        "expected_contains": ["santoku"],
        "expected_not_contains": ["perhaps", "you might also consider", "alternatively",
                                  "if you want", "whenever you're ready"],
        "notes": "Purchase-ready + context signals — Santoku. Decisive. No hedging."
    },
    {
        "id": "B04",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "Can I order the Bread Knife and get it delivered by the weekend?",
        "intent": "purchase_intent",
        "trap": "Vague: 'We'll do our best to accommodate your timeline! Our team will look into this.'",
        "expected_contains": [],
        "expected_not_contains": ["we'll do our best", "we cannot guarantee",
                                  "we apologize", "please allow", "rest assured"],
        "notes": "Time-sensitive purchase — direct answer, no vague promises."
    },
    {
        "id": "B05",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "Do you ship internationally? I'm in Germany.",
        "intent": "purchase_intent",
        "trap": "Inventing shipping details or overly corporate language.",
        "expected_contains": [],
        "expected_not_contains": ["we apologize", "unfortunately we", "valued customer",
                                  "we strive to", "please do not hesitate"],
        "notes": "Logistical question — don't invent facts. Brand voice redirect to website."
    },
    {
        "id": "B06",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard", "context"],
        "message": "I'm buying this as a gift for someone who loves to cook but always uses bad knives.",
        "intent": "purchase_intent",
        "trap": "Over-explaining or using gift-wrap corporate language.",
        "expected_contains": ["artisan chef", "beginner set"],
        "expected_not_contains": ["excellent choice", "they're going to love it",
                                  "is there anything else", "let me know if"],
        "notes": "Gift purchase — decisive recommendation. No gushing. No corporate close."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 6: EDGE CASES (6 cases)
    # All three modules stressed simultaneously — hardest test set
    # ═══════════════════════════════════════════════════════════════════,
    {
        "id": "X01",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "I'm so overwhelmed in the kitchen. Everything takes me so long and nothing comes out right. I don't know if this is for me.",
        "intent": "fear_concern",
        "trap": "Platitude: 'Don't give up! You've got this! Every chef starts somewhere!'",
        "expected_contains": [],
        "expected_not_contains": ["you've got this", "don't give up", "believe in yourself",
                                  "every chef starts somewhere", "amazing journey"],
        "notes": "Defeated + fear — tone=support, context detects defeated emotion, guard blocks platitudes."
    },
    {
        "id": "X02",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "I want a refund AND I'm going to sue AND I'll be buying from Wusthof from now on.",
        "intent": "return_refund",
        "trap": "Addressing all three simultaneously with panic, or mentioning Wusthof.",
        "expected_contains": ["not right"],
        "expected_not_contains": ["wusthof", "legal", "lawsuit", "we understand",
                                  "we apologize for any inconvenience"],
        "notes": "Legal + refund + competitor — own it, fix it, silence on the rest."
    },
    {
        "id": "X03",
        "bucket": "edge_case",
        "primary_stress": ["context", "brand_guard"],
        "message": "Your knives are beautiful and I love them! But also — my Santoku chipped on a carrot.",
        "intent": "product_issue",
        "trap": "Getting swept up in the praise, or using corporate close after handling the complaint.",
        "expected_contains": ["not right"],
        "expected_not_contains": ["thank you so much", "we're delighted",
                                  "does that make sense", "is there anything else"],
        "notes": "Praise + complaint — brief warmth, then direct accountability. No corporate close."
    },
    {
        "id": "X04",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "I'm a culinary school instructor and I recommend knives to students every year. What would you suggest for a beginner curriculum?",
        "intent": "product_question",
        "trap": "Treating as consumer beginner query — wrong recommendation and tone.",
        "expected_contains": ["paring knife", "beginner set"],
        "expected_not_contains": ["you've got this", "where do you start",
                                  "hrc", "rockwell", "blade angle"],
        "notes": "Professional educator context — Context detects professional intent. Practical and direct."
    },
    {
        "id": "X05",
        "bucket": "edge_case",
        "primary_stress": ["context", "brand_guard"],
        "message": "I bake all my own sourdough AND I do a lot of fine vegetable prep. What's my priority knife?",
        "intent": "product_question",
        "trap": "Recommending Bread Knife only (missing vegetable prep signal) OR listing both.",
        "expected_contains": ["artisan chef"],
        "expected_not_contains": ["both the", "either the", "you could get both",
                                  "alternatively", "does that make sense"],
        "notes": "Conflicting signals (bread vs veg) — Artisan Chef's Knife handles most cases. Single answer."
    },
    {
        "id": "X06",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "My knife broke, I sliced my finger, I've had to cancel my dinner party, and I'm devastated.",
        "intent": "product_issue",
        "trap": "Either ignoring the injury for the complaint or ignoring the complaint for the injury.",
        "expected_contains": ["not right"],
        "expected_not_contains": ["we apologize for any inconvenience", "sorry to hear that",
                                  "you've got this", "sensory", "whisper"],
        "notes": "Product failure + injury + emotional distress. Own defect first. No poetry. No platitudes."
    },
]

# ── Metadata ───────────────────────────────────────────────────────────────────

BUCKET_SUMMARY = {
    "complaint":       {"count": 6, "primary_stress": "Brand Guard + Context",
                        "ablation_claim": "Conditions A/B score low; C/D improve tone but miss accountability; E/F highest"},
    "escalation":      {"count": 6, "primary_stress": "Context + Brand Guard",
                        "ablation_claim": "Condition A/C miss escalation signals entirely; B detects but no guard constraints"},
    "competitor":      {"count": 6,  "primary_stress": "Brand Guard + Context",
                        "ablation_claim": "A/B/C all risk competitor mentions; only E/F reliably block + redirect to craft"},
    "product_question":{"count": 6,  "primary_stress": "Context",
                        "ablation_claim": "A/C give wrong knife; B/D/E/F give correct knife from cook-type signals"},
    "purchase_intent": {"count": 6,  "primary_stress": "Brand Guard",
                        "ablation_claim": "A/B/C/D may use pushy or corporate language; E/F suppress it"},
    "edge_case":       {"count": 6,  "primary_stress": "All three modules",
                        "ablation_claim": "Only F (full system) handles all dimensions correctly"},
}

TOTAL_CASES = len(TEST_CASES)


def get_cases_by_bucket(bucket: str):
    return [t for t in TEST_CASES if t["bucket"] == bucket]


def get_cases_by_stress(module: str):
    return [t for t in TEST_CASES if module in t["primary_stress"]]


def get_case_by_id(case_id: str):
    return next((t for t in TEST_CASES if t["id"] == case_id), None)


if __name__ == "__main__":
    print(f"Total test cases: {TOTAL_CASES}")
    print()
    for bucket, meta in BUCKET_SUMMARY.items():
        cases = get_cases_by_bucket(bucket)
        print(f"  {bucket:20s} {meta['count']:3d} cases | stress: {meta['primary_stress']}")
    print()
    print("Cases stressing context_engine :", len(get_cases_by_stress("context")))
    print("Cases stressing brand_guard    :", len(get_cases_by_stress("brand_guard")))
    print("Cases stressing tone           :", len(get_cases_by_stress("tone")))