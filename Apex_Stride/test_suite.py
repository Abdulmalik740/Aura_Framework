"""
Test Suite — EMNLP 2026 Ablation Study
=======================================
48 scenarios across 6 buckets, designed to maximally stress
each module so ablation deltas are large and interpretable.

Bucket design rationale:
  COMPLAINTS (10)      → Brand Guard critical: ownership language, no corporate apology
  ESCALATION (10)      → Context critical: detect legal/anger signals, guard prevents meltdown
  COMPETITOR (8)       → Guard + Context together: philosophy redirect, no feature compare
  PRODUCT/SIZING (8)   → Context critical: right product from athlete profile
  PURCHASE INTENT (6)  → Guard critical: no pushy language, facts only
  EDGE CASES (6)       → All three modules stressed simultaneously

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

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 1: COMPLAINTS + PRODUCT ISSUES (10 cases)
    # Primary stress: Brand Guard (ownership language, no corporate apology)
    # Without guard → "we apologize for any inconvenience", "valued customer"
    # Without context → no intent detection, won't trigger ownership behaviour
    # ═══════════════════════════════════════════════════════════════════

    {
        "id": "C01",
        "bucket": "complaint",
        "primary_stress": ["brand_guard", "context"],
        "message": "My Runner 5s fell apart after just 6 weeks. The sole completely detached on my morning run.",
        "intent": "product_issue",
        "trap": "Generic apology: 'We apologize for any inconvenience caused.'",
        "expected_contains": ["on us", "order number", "photo"],
        "expected_not_contains": ["apologize for any inconvenience", "valued customer",
                                   "sorry to hear", "we understand your frustration"],
        "notes": "Classic product defect. Guard must enforce 'That's on us' not corporate apology."
    },
    {
        "id": "C02",
        "bucket": "complaint",
        "primary_stress": ["brand_guard", "context"],
        "message": "These shoes are the worst I've ever bought. Terrible quality, total waste of money.",
        "intent": "complaint",
        "trap": "Lengthy empathetic explanation with weak language: 'We're sorry you feel that way...'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we apologize", "sorry you feel", "we understand",
                                   "inconvenience", "is there anything else"],
        "notes": "Angry complaint — context detects anger, guard enforces direct ownership."
    },
    {
        "id": "C03",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "The stitching on my Velocity X2 is already coming undone and I've only had them 2 months.",
        "intent": "product_issue",
        "trap": "Corporate: 'Thank you for bringing this to our attention, valued customer.'",
        "expected_contains": ["on us", "photo"],
        "expected_not_contains": ["thank you for contacting", "valued customer",
                                   "bringing this to our attention"],
        "notes": "Stitching defect — ownership + resolution path required."
    },
    {
        "id": "C04",
        "bucket": "complaint",
        "primary_stress": ["context", "brand_guard"],
        "message": "I trained 6 months for my half marathon and your shoes gave me blisters on race day. I'm devastated.",
        "intent": "product_issue",
        "trap": "Generic empathy without specific acknowledgment of the 6-month investment.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["sorry to hear that", "we appreciate your feedback",
                                   "we apologize for any inconvenience"],
        "notes": "Emotional weight — context detects 'injured/disappointed', guard enforces ownership."
    },
    {
        "id": "C05",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "Just got my order and the wrong size was shipped. I needed these for tomorrow's race.",
        "intent": "order_inquiry",
        "trap": "Passive: 'We will look into this matter and someone will be in touch.'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we'll look into it", "someone will be in touch",
                                   "we'll get back to you", "apologize for any inconvenience"],
        "notes": "Urgent wrong order — time sensitivity + direct fix required."
    },
    {
        "id": "C06",
        "bucket": "complaint",
        "primary_stress": ["brand_guard", "context"],
        "message": "Your Marathon Pro destroyed my knees. I can barely walk today after my long run.",
        "intent": "complaint",
        "trap": "Recommending Recovery Stride before owning the issue — wrong priority.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we apologize", "sorry to hear", "inconvenience",
                                   "is there anything else I can help"],
        "notes": "Injury complaint — must own first, then recovery action. No product push."
    },
    {
        "id": "C07",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "I want a refund. These shoes are falling apart and I've only worn them 4 times.",
        "intent": "return_refund",
        "trap": "Lengthy refund process explanation without owning the defect first.",
        "expected_contains": ["on us", "order number"],
        "expected_not_contains": ["we apologize for any inconvenience", "please don't hesitate",
                                   "thank you for contacting", "dear valued customer"],
        "notes": "Direct refund + complaint context — ownership before process."
    },
    {
        "id": "C08",
        "bucket": "complaint",
        "primary_stress": ["context", "brand_guard"],
        "message": "I specifically asked customer service last week about sizing and followed their advice. Now they don't fit.",
        "intent": "complaint",
        "trap": "Deflecting: 'I'd recommend reaching out to our customer service team.'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["reach out to", "contact customer service",
                                   "get in touch with our team", "please contact us"],
        "notes": "Company error — clear ownership, no deflection to another team."
    },
    {
        "id": "C09",
        "bucket": "complaint",
        "primary_stress": ["brand_guard"],
        "message": "I need a replacement pair. My Ultra Trail shoes are completely worn out after 3 months of trail running.",
        "intent": "replacement",
        "trap": "Corporate: 'We appreciate your loyalty and will process your replacement request.'",
        "expected_contains": ["on us", "order number", "photo"],
        "expected_not_contains": ["appreciate your loyalty", "we value your business",
                                   "dear customer", "we apologize for any inconvenience"],
        "notes": "Replacement request with complaint context — ownership + immediate resolution."
    },
    {
        "id": "C10",
        "bucket": "complaint",
        "primary_stress": ["brand_guard", "context"],
        "message": "The cushioning on these is nothing like advertised. False advertising!",
        "intent": "complaint",
        "trap": "Defensive: 'Our products are designed to meet the highest standards...'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we apologize", "our products are designed",
                                   "we strive to", "we understand your concerns"],
        "notes": "False advertising claim — direct ownership, no defensive justification."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 2: ESCALATION / ANGRY USERS (10 cases)
    # Primary stress: Context (detects escalation signals) + Guard (no panic language)
    # Without context → escalation signals missed, no adjusted behaviour
    # Without guard → may produce corporate panic response
    # ═══════════════════════════════════════════════════════════════════

    {
        "id": "E01",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "I'm going to sue your company if you don't fix this immediately. My shoes fell apart and I twisted my ankle.",
        "intent": "product_issue",
        "trap": "Acknowledging the lawsuit threat directly: 'We take legal matters very seriously...'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["legal", "lawyer", "lawsuit", "we take this seriously",
                                   "escalate to our legal team", "apologize for any inconvenience"],
        "notes": "Legal threat — context detects escalation + injury. Guard silences legal thread."
    },
    {
        "id": "E02",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "This is completely unacceptable. I'm filing a complaint with the BBB right now.",
        "intent": "complaint",
        "trap": "Panicked corporate response: 'We sincerely apologize and want to resolve this immediately...'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["bbb", "better business", "sincerely apologize",
                                   "we take all feedback seriously", "valued customer"],
        "notes": "BBB threat — same response as any complaint: own it, fix it. Silence on threat."
    },
    {
        "id": "E03",
        "bucket": "escalation",
        "primary_stress": ["context"],
        "message": "WORST EXPERIENCE EVER. Your shoes destroyed my knee during a race I've been training for all year. I'm FURIOUS.",
        "intent": "product_issue",
        "trap": "Missing the emotional intensity — treating it like a generic product question.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we understand your frustration", "we apologize for",
                                   "we're sorry to hear", "is there anything else"],
        "notes": "High-intensity anger — context must detect intensity 5, guard enforces ownership."
    },
    {
        "id": "E04",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "I want to speak to a manager. These shoes are a scam and I'm going to tell everyone I know.",
        "intent": "complaint",
        "trap": "Escalating to manager offer: 'I'd be happy to connect you with our management team...'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["connect you with", "transfer you to", "manager",
                                   "supervisor", "escalate your concern", "we apologize"],
        "notes": "Manager demand — context detects anger, guard prevents deflection."
    },
    {
        "id": "E05",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "I've contacted you 3 times about this refund and nothing has been done. I'm contacting my credit card company for a chargeback.",
        "intent": "return_refund",
        "trap": "Generic: 'We're sorry for the delay. Please allow 5-7 business days...'",
        "expected_contains": ["on us", "order number"],
        "expected_not_contains": ["please allow", "business days", "we apologize for the delay",
                                   "we understand your frustration", "rest assured"],
        "notes": "Churn risk + escalation — context detects churn signals, fast ownership required."
    },
    {
        "id": "E06",
        "bucket": "escalation",
        "primary_stress": ["context"],
        "message": "Your company is a fraud. I'm going to post about this on every social media platform I can find.",
        "intent": "complaint",
        "trap": "Panicked PR response: 'We take your concerns very seriously and would like to resolve this...'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we take your concerns", "we take this very seriously",
                                   "reputation", "social media", "apologize"],
        "notes": "Fraud accusation — silence on threat, own the issue, fix it."
    },
    {
        "id": "E07",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "My lawyer will be hearing about this. The sole detached mid-race and I fell and injured myself.",
        "intent": "product_issue",
        "trap": "Defensive legal language or excessive apology.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["legal", "lawyer", "attorney", "liability",
                                   "we apologize for any inconvenience", "sincerely sorry"],
        "notes": "Legal + injury — one sentence owning specific failure. Silence on legal."
    },
    {
        "id": "E08",
        "bucket": "escalation",
        "primary_stress": ["context"],
        "message": "I am absolutely disgusted. This is the second pair that's fallen apart. Done with Apex forever.",
        "intent": "complaint",
        "trap": "Missing churn signal and churn_risk not elevated.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we apologize", "thank you for your loyalty",
                                   "we hope you'll give us another chance", "valued customer"],
        "notes": "Repeat issue + churn signal — context must detect churn, no retention platitudes."
    },
    {
        "id": "E09",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "What is wrong with your company? I ordered 2 weeks ago, no delivery, no tracking update, no response from support.",
        "intent": "order_inquiry",
        "trap": "Passive: 'We apologize for the inconvenience. Please allow additional processing time.'",
        "expected_contains": ["on us"],
        "expected_not_contains": ["allow additional", "processing time", "apologize for the inconvenience",
                                   "please be patient", "we are working on it"],
        "notes": "Angry order inquiry — urgency 5, context detects frustration, ownership first."
    },
    {
        "id": "E10",
        "bucket": "escalation",
        "primary_stress": ["context", "brand_guard"],
        "message": "This is ridiculous. I paid $150 for these shoes and they're already trash after one month. I want my money back NOW.",
        "intent": "return_refund",
        "trap": "Starting with process steps before owning the failure.",
        "expected_contains": ["on us", "order number"],
        "expected_not_contains": ["we apologize for any inconvenience", "sorry for",
                                   "please allow", "our return policy states", "valued customer"],
        "notes": "Angry refund demand — ownership before process, no policy language."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 3: COMPETITOR COMPARISONS (8 cases)
    # Primary stress: Brand Guard (never name competitors, You vs You)
    #                 Context (detect competition_comparison intent)
    # Without guard → may name competitors or compare features
    # Without context → won't detect comparison intent, gives product pitch instead
    # ═══════════════════════════════════════════════════════════════════

    {
        "id": "P01",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "Why should I buy from you instead of Nike? They have way better technology.",
        "intent": "competition_comparison",
        "trap": "Engaging with technology comparison or mentioning Nike in response.",
        "expected_contains": ["you vs"],
        "expected_not_contains": ["nike", "technology", "better than", "outperform",
                                   "we also", "our technology"],
        "notes": "Direct challenge — guard must block Nike mention, tone opens with You vs You."
    },
    {
        "id": "P02",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "My running group all swears by Brooks. I feel like the odd one out buying Apex Stride.",
        "intent": "competition_comparison",
        "trap": "Validating Brooks or group pressure, or trying to pitch Apex Stride features to fit in.",
        "expected_contains": ["you vs"],
        "expected_not_contains": ["brooks", "popular", "your group", "understand",
                                   "a lot of runners", "great choice"],
        "notes": "Social pressure framing — B may empathise with group pressure and forget philosophy."
    },
    {
        "id": "P03",
        "bucket": "competitor",
        "primary_stress": ["context", "brand_guard"],
        "message": "Adidas Ultraboost is on sale for $80 less than your equivalent shoe. Hard to justify the price difference.",
        "intent": "competition_comparison",
        "trap": "Justifying price with features, comparing value, or mentioning Adidas.",
        "expected_contains": ["you vs"],
        "expected_not_contains": ["adidas", "price", "worth it", "value", "investment",
                                   "quality difference", "you get what you pay for"],
        "notes": "Price challenge — hardest for B, LLM naturally wants to justify price with features."
    },
    {
        "id": "P04",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "These shoes destroyed my feet on a long run. My old Asics never did this. Switching back.",
        "intent": "competition_comparison",
        "trap": "Getting into comparison with Asics OR not owning the product failure first.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["asics", "better fit", "more cushioning",
                                   "different from", "compared to asics"],
        "notes": "Complaint + competitor — must own failure first (OC), then no comparison (CCS). Hardest case."
    },
    {
        "id": "P05",
        "bucket": "competitor",
        "primary_stress": ["brand_guard", "context"],
        "message": "My friend swears by New Balance for recovery runs. Are yours as good?",
        "intent": "competition_comparison",
        "trap": "Confirming or denying the comparison, or mentioning New Balance positively.",
        "expected_contains": ["you vs"],
        "expected_not_contains": ["new balance", "as good as", "better than",
                                   "comparable", "similar"],
        "notes": "Indirect comparison — you vs you redirect + Recovery Stride mention acceptable."
    },
    {
        "id": "P06",
        "bucket": "competitor",
        "primary_stress": ["context", "brand_guard"],
        "message": "I'm a beginner runner. Everyone tells me to just get Nikes. Why shouldn't I?",
        "intent": "competition_comparison",
        "trap": "Mentioning Nike or launching into feature pitch for Runner 5.",
        "expected_contains": ["you vs"],
        "expected_not_contains": ["nike", "everyone starts", "better option",
                                   "our runner 5 is better", "compared to nike"],
        "notes": "Beginner + competitor — welcome without competing, philosophy first."
    },
    {
        "id": "P07",
        "bucket": "competitor",
        "primary_stress": ["brand_guard"],
        "message": "Adidas Ultraboost is cheaper and looks better. Change my mind.",
        "intent": "competition_comparison",
        "trap": "Engaging with price or aesthetics comparison.",
        "expected_contains": ["you vs"],
        "expected_not_contains": ["adidas", "ultraboost", "cheaper", "price",
                                   "looks", "aesthetics", "design"],
        "notes": "Price + style bait — guard blocks competitor name and feature trap."
    },
    {
        "id": "P08",
        "bucket": "competitor",
        "primary_stress": ["context", "brand_guard"],
        "message": "I'm angry. I spent $200 on your shoes and they broke. A $60 pair of Skechers lasted longer.",
        "intent": "complaint",
        "trap": "Treating as comparison and not owning the product failure.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["skechers", "compared to", "price point",
                                   "we apologize for any inconvenience"],
        "notes": "Angry complaint with competitor mention — intent is complaint not comparison. Own it."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 4: PRODUCT QUESTIONS + SIZING (8 cases)
    # Primary stress: Context (right product from athlete profile signals)
    # Without context → random or default product recommendation
    # Without guard → may mention competitors or use platitudes
    # ═══════════════════════════════════════════════════════════════════

    {
        "id": "Q01",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I just started running last month and want to do my first 5K. Which shoe?",
        "intent": "product_question",
        "trap": "Recommending Velocity X2 (race shoe for beginners is wrong).",
        "expected_contains": ["runner 5"],
        "expected_not_contains": ["velocity x2", "marathon pro", "ultra trail",
                                   "you got this", "discover your potential"],
        "notes": "Beginner — context must detect beginner athlete type → Runner 5."
    },
    {
        "id": "Q02",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "Training for a competitive 10K. Sub-45 is my goal. What shoe?",
        "intent": "product_question",
        "trap": "Recommending Runner 5 (wrong for competitive speed work).",
        "expected_contains": ["velocity x2"],
        "expected_not_contains": ["runner 5", "marathon pro",
                                   "you got this", "crush it"],
        "notes": "Competitive 10K — context detects competitive athlete + speed intent → Velocity X2."
    },
    {
        "id": "Q03",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I'm recovering from shin splints and want to get back to running carefully.",
        "intent": "product_question",
        "trap": "Recommending Velocity X2 or Marathon Pro for injury recovery.",
        "expected_contains": ["recovery stride"],
        "expected_not_contains": ["velocity x2", "marathon pro",
                                   "you got this", "push through the pain"],
        "notes": "Injury recovery — context detects injury + comeback → Recovery Stride."
    },
    {
        "id": "Q04",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I run ultra marathons and need something for technical rocky trails.",
        "intent": "product_question",
        "trap": "Recommending Marathon Pro (wrong — no trail grip).",
        "expected_contains": ["ultra trail"],
        "expected_not_contains": ["marathon pro", "runner 5",
                                   "discover your potential", "test them out"],
        "notes": "Elite trail runner — context detects elite + ultra context → Ultra Trail."
    },
    {
        "id": "Q05",
        "bucket": "product_question",
        "primary_stress": ["context"],
        "message": "I run every morning at 5am before work. Need something for daily 10K training.",
        "intent": "product_question",
        "trap": "Recommending race shoe instead of daily trainer.",
        "expected_contains": ["runner 5"],
        "expected_not_contains": ["velocity x2", "ultra trail",
                                   "find what feels right", "look for shoes with"],
        "notes": "5am club daily runner — context detects 5am_club athlete type → Runner 5."
    },
    {
        "id": "Q06",
        "bucket": "product_question",
        "primary_stress": ["context", "brand_guard"],
        "message": "I'm training for the Boston Marathon qualifying attempt. 3:05 is my target.",
        "intent": "product_question",
        "trap": "Generic marathon shoe rec without acknowledging competitive context.",
        "expected_contains": ["marathon pro", "velocity x2"],
        "expected_not_contains": ["runner 5", "you got this",
                                   "discover your potential", "good luck"],
        "notes": "BQ attempt — non-beginner marathon → Marathon Pro or Velocity X2."
    },
    {
        "id": "Q07",
        "bucket": "product_question",
        "primary_stress": ["brand_guard"],
        "message": "What size should I get? I'm normally an 11 in most brands.",
        "intent": "sizing_help",
        "trap": "Generic: 'We recommend trying them on in store to find what feels right.'",
        "expected_contains": [],
        "expected_not_contains": ["find what feels right", "test them out",
                                   "try them on", "look for shoes with",
                                   "good support and cushioning"],
        "notes": "Sizing — guard blocks 'find what feels right' and 'test them out' platitudes."
    },
    {
        "id": "Q08",
        "bucket": "product_question",
        "primary_stress": ["context", "brand_guard"],
        "message": "Coming back after ACL surgery, 8 months out. Need the right shoe for my comeback.",
        "intent": "product_question",
        "trap": "Recommending Velocity X2 or treating as regular product question.",
        "expected_contains": ["recovery stride"],
        "expected_not_contains": ["velocity x2", "marathon pro",
                                   "you got this", "you'll be back in no time"],
        "notes": "ACL comeback — context detects comeback athlete → Recovery Stride."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 5: PURCHASE INTENT (6 cases)
    # Primary stress: Brand Guard (no pushy language: limited time, act now, hurry)
    # Without guard → pushy sales language likely
    # Without context → may over-explain instead of state facts + imperative
    # ═══════════════════════════════════════════════════════════════════

    {
        "id": "B01",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "Is the Runner 5 in stock? I want to order today.",
        "intent": "purchase_intent",
        "trap": "Pushy: 'Order now — limited stock available! Don't miss out!'",
        "expected_contains": [],
        "expected_not_contains": ["limited time", "don't miss out", "act now",
                                   "hurry", "selling fast", "order now before"],
        "notes": "Direct purchase — facts only, no urgency manufacturing."
    },
    {
        "id": "B02",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "How much is the Velocity X2?",
        "intent": "purchase_intent",
        "trap": "Sales pitch: 'At just $X, it's an incredible value for the performance you get!'",
        "expected_contains": [],
        "expected_not_contains": ["incredible value", "great deal", "worth every penny",
                                   "limited time", "act now", "don't miss"],
        "notes": "Price question — state fact, no value sell."
    },
    {
        "id": "B03",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard", "context"],
        "message": "I'm ready to buy the Marathon Pro. Any discount codes?",
        "intent": "purchase_intent",
        "trap": "Over-promising on discounts or using corporate language.",
        "expected_contains": [],
        "expected_not_contains": ["we apologize", "unfortunately we don't",
                                   "valued customer", "exclusive offer", "special deal for you"],
        "notes": "Discount ask — factual response in brand voice, no corporate apology."
    },
    {
        "id": "B04",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "Can I order the Ultra Trail and get it by next weekend?",
        "intent": "purchase_intent",
        "trap": "Vague: 'We cannot guarantee delivery times but we'll do our best!'",
        "expected_contains": [],
        "expected_not_contains": ["we'll do our best", "we cannot guarantee",
                                   "we apologize", "please allow", "rest assured"],
        "notes": "Time-sensitive purchase — direct answer, no vague promises."
    },
    {
        "id": "B05",
        "bucket": "purchase_intent",
        "primary_stress": ["context", "brand_guard"],
        "message": "I want to buy shoes but I'm not sure which one. I run 3x a week casually.",
        "intent": "purchase_intent",
        "trap": "Listing all products or using 'find what feels right' language.",
        "expected_contains": ["runner 5"],
        "expected_not_contains": ["find what feels right", "test them out",
                                   "you might want to consider", "perhaps"],
        "notes": "Casual runner + purchase intent — context → Runner 5, guard blocks platitudes."
    },
    {
        "id": "B06",
        "bucket": "purchase_intent",
        "primary_stress": ["brand_guard"],
        "message": "Do you ship internationally? I'm in Australia.",
        "intent": "purchase_intent",
        "trap": "Inventing shipping info or using vague corporate language.",
        "expected_contains": [],
        "expected_not_contains": ["we apologize", "unfortunately", "please note",
                                   "valued customer", "we strive to", "we do our best"],
        "notes": "Logistical question — don't invent facts. Website redirect in brand voice."
    },

    # ═══════════════════════════════════════════════════════════════════
    # BUCKET 6: EDGE CASES (6 cases)
    # All three modules stressed simultaneously — hardest test set
    # ═══════════════════════════════════════════════════════════════════

    {
        "id": "X01",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "I feel like quitting running altogether. What's even the point? I keep getting injured.",
        "intent": "motivation_seeking",
        "trap": "Platitude: 'You've got this! Every step counts! Believe in yourself!'",
        "expected_contains": [],
        "expected_not_contains": ["you got this", "you've got this", "believe in yourself",
                                   "every step counts", "don't give up", "you're amazing"],
        "notes": "Defeated + motivation — tone=support, context detects defeated emotion, guard blocks platitudes."
    },
    {
        "id": "X02",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "I want to sue you AND I want a refund AND I'm switching to Nike.",
        "intent": "return_refund",
        "trap": "Addressing all three simultaneously with panic or mentioning Nike.",
        "expected_contains": ["on us", "order number"],
        "expected_not_contains": ["nike", "legal", "lawyer", "we understand",
                                   "we apologize for any inconvenience"],
        "notes": "Legal + refund + competitor — own it, fix it, silence on rest."
    },
    {
        "id": "X03",
        "bucket": "edge_case",
        "primary_stress": ["context", "brand_guard"],
        "message": "Your shoes are amazing! But also, can I get them in wide width?",
        "intent": "product_question",
        "trap": "Excessive praise + unhelpful sizing answer with platitudes.",
        "expected_contains": [],
        "expected_not_contains": ["thank you for contacting", "so glad you love",
                                   "find what feels right", "test them out"],
        "notes": "Praise + product question — brief acknowledgment, direct sizing answer."
    },
    {
        "id": "X04",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "I'm a physical therapist recommending shoes to my patients recovering from foot surgery. What do you suggest?",
        "intent": "product_question",
        "trap": "Treating as consumer query — wrong athlete profile, wrong recommendation.",
        "expected_contains": ["recovery stride"],
        "expected_not_contains": ["velocity x2", "marathon pro",
                                   "you got this", "lace up and go"],
        "notes": "Professional/clinical context — context must detect injury/recovery, tone should be slightly more informational."
    },
    {
        "id": "X05",
        "bucket": "edge_case",
        "primary_stress": ["context", "brand_guard"],
        "message": "I train at 5am every day and I'm training for an ultra marathon in the mountains. Shoe rec?",
        "intent": "product_question",
        "trap": "Recommending Runner 5 (5am club keyword) when Ultra Trail is clearly right.",
        "expected_contains": ["ultra trail"],
        "expected_not_contains": ["runner 5", "marathon pro",
                                   "find what feels right", "test them out"],
        "notes": "Conflicting signals (5am_club vs ultra) — context must prioritise trail/ultra context."
    },
    {
        "id": "X06",
        "bucket": "edge_case",
        "primary_stress": ["context", "tone", "brand_guard"],
        "message": "My shoes broke, I'm injured, I haven't run in 3 months, I'm losing fitness and I'm angry at everything.",
        "intent": "product_issue",
        "trap": "Either ignoring the injury for the complaint or ignoring the complaint for the injury.",
        "expected_contains": ["on us"],
        "expected_not_contains": ["we apologize for any inconvenience", "you got this",
                                   "believe in yourself", "every step counts", "sorry to hear"],
        "notes": "Multi-issue: product issue + injury + emotional distress. Own the defect first, then recovery path."
    },
]

# ── Metadata ──────────────────────────────────────────────────────────────────

BUCKET_SUMMARY = {
    "complaint":       {"count": 10, "primary_stress": "Brand Guard + Context",
                        "ablation_claim": "Conditions A/B score low; C/D improve tone but miss ownership; E/F highest"},
    "escalation":      {"count": 10, "primary_stress": "Context + Brand Guard",
                        "ablation_claim": "Condition A/C miss escalation signals entirely; B detects but no guard constraints"},
    "competitor":      {"count": 8,  "primary_stress": "Brand Guard + Context",
                        "ablation_claim": "A/B/C all risk competitor mentions; only E/F reliably block"},
    "product_question":{"count": 8,  "primary_stress": "Context",
                        "ablation_claim": "A/C give wrong product; B/D/E/F give correct product"},
    "purchase_intent": {"count": 6,  "primary_stress": "Brand Guard",
                        "ablation_claim": "A/B/C/D may use pushy language; E/F suppress it"},
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