# tests/test_stage1_realistic_scores.py
"""
Stage-1 realistic examples test (multi-label + scores).

- Imports classify_content_type (multi_label=True, return_scores=True).
- Uses medium-hard, realistic examples (messy, observational descriptions).
- Prints per-example scored outputs and a simple pass/fail summary.
"""

import pprint
pp = pprint.PrettyPrinter(indent=2)

# Try project import path first, then fallback to local
try:
    from pugsy_ai.pipelines.fastvlm.stage1_classifier import classify_content_type, CATEGORY_DESCRIPTIONS
except Exception:
    try:
        from stage1_classifier import classify_content_type, CATEGORY_DESCRIPTIONS
    except Exception as e:
        raise ImportError("Couldn't import stage1_classifier. Error: " + str(e))


TEST_CASES = {
    "lifestyle": [
        "A woman is sitting on a wooden bench outside a building, wearing a white knitted sweater and holding a smartphone while casually scrolling. A brown leather handbag is placed beside her. The overall vibe feels like someone relaxing outdoors during a break.",
        "A guy is sitting at his kitchen table eating cereal while his laptop is open in front of him. The room looks lived-in, with plants on the shelf and a half-open window behind him letting in morning light.",
    ],
    "fashion_beauty": [
        "A woman poses on a narrow street wearing an oversized beige coat, knee-high boots, and holding a small black purse. She's adjusting her sunglasses while the wind moves her hair slightly.",
        "A creator is applying foundation with a beauty blender in front of a bright ring light. Multiple makeup products are spread across the table.",
    ],
    "fitness_sports": [
        "A man is doing dumbbell shoulder presses at the gym while droplets of sweat roll down his forehead. Several weight racks and gym mirrors are visible behind him.",
        "A runner adjusts their shoelaces before starting a jog on a park trail early in the morning. There's slight fog in the background.",
    ],
    "travel_nature": [
        "Someone walking through a busy foreign market while narrating their trip.",
        "A person is sitting on the edge of a cliff overlooking a valley full of green hills. The wind blows their jacket, and clouds drift slowly in the sky.",
    ],
    "food_cooking": [
        "A chef drizzles olive oil over chopped tomatoes and basil on a cutting board. The kitchen is messy, with open spice jars in the background.",
        "A street vendor is flipping flatbreads quickly over a sizzling pan while customers wait in line.",
    ],
    "comedy_memes": [
        "A guy spills coffee on himself right before stepping out the door, immediately looks into the camera with a 'why me?' face.",
        "A dog tries to jump onto a couch, fails, slides backward, and walks away pretending nothing happened.",
    ],
}

TOP_K = 3
SCORE_THRESHOLD = 0.30  # used by classifier for filtering when multi_label=True

def tolerant_match(preds_with_scores, expected_key):
    """
    Return True if expected_key is among predicted labels OR
    if any predicted label's description contains a keyword from expected_key.
    preds_with_scores: [(label, score), ...]
    """
    if not preds_with_scores:
        return False
    labels = [p for p, s in preds_with_scores]
    if expected_key in labels:
        return True
    exp_words = expected_key.split("_")
    for lab, _ in preds_with_scores:
        desc = CATEGORY_DESCRIPTIONS.get(lab, "").lower()
        if any(w in desc for w in exp_words):
            return True
    return False

def run_tests():
    total = 0
    fails = []
    print("Running Stage-1 realistic multi-label tests (top_k=%d)..." % TOP_K)
    print("Note: classifier called with multi_label=True, return_scores=True, score_threshold=%s\n" % SCORE_THRESHOLD)

    for cat, examples in TEST_CASES.items():
        print("=" * 80)
        print("Category (expected):", cat)
        for ex in examples:
            total += 1
            try:
                preds = classify_content_type(
                    ex,
                    top_k=TOP_K,
                    use_zero_shot=True,
                    score_threshold=SCORE_THRESHOLD,
                    multi_label=True,
                    return_scores=True,
                )
            except Exception as e:
                print("  ERROR calling classifier:", e)
                preds = []

            print("\nExample:", ex)
            print("Predictions (label, score):")
            pp.pprint(preds)

            ok = tolerant_match(preds, cat)
            print("Match OK:", ok)
            if not ok:
                fails.append((cat, ex, preds))

    print("\n" + "=" * 80)
    print("Total examples:", total)
    print("Failures:", len(fails))
    if fails:
        print("\nFailed cases:")
        pp.pprint(fails)
        raise SystemExit(f"{len(fails)} tests failed")
    else:
        print("ALL TESTS PASSED ✅")

if __name__ == "__main__":
    run_tests()
