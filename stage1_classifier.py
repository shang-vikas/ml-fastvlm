# stage1_classifier.py
"""
Stage-1 content classifier (multi-label + scores, zero-shot + keyword fallback).

New features:
 - multi_label support (transformers pipeline multi_label=True)
 - optional return_scores flag to get (label, score) tuples
 - top_k controls number of returned labels
 - score_threshold filters out low-confidence labels (applies to multi-label)
 - still falls back to keyword-based scoring when pipeline missing

API:
    classify_content_type(
        summary: str,
        top_k: int = 1,
        use_zero_shot: bool = True,
        score_threshold: float = 0.30,
        multi_label: bool = False,
        return_scores: bool = False
    ) -> List[str] or List[Tuple[str, float]]

Backward compatible: default returns List[str] (single label).
"""
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger("fastvlm.stage1")
logger.addHandler(logging.NullHandler())

# Canonical category list (short keys)
CATEGORY_LIST = [
    "lifestyle",
    "fashion_beauty",
    "fitness_sports",
    "food_cooking",
    "travel_nature",
    "comedy_memes",
    "pets_animals",
    "music_dance",
    "art_creativity",
    "technology_gadgets",
    "education_tips",
    "business_motivation",
]

# One-line descriptors (for docs)
CATEGORY_DESCRIPTIONS = {
    "lifestyle": "Everyday personal moments, routines, home life, or casual social experiences.",
    "fashion_beauty": "Content focused on makeup, skincare, hairstyles, outfits, styling, or personal appearance.",
    "fitness_sports": "Activities involving exercise, workouts, physical training, or sports performance.",
    "food_cooking": "Videos showing food, recipes, cooking steps, restaurants, or eating experiences.",
    "travel_nature": "Scenes of destinations, landscapes, outdoor exploration, or travel experiences.",
    "comedy_memes": "Humorous skits, jokes, reactions, relatable moments, or playful meme-style content.",
    "pets_animals": "Content featuring domestic pets or animals in natural or playful settings.",
    "music_dance": "Performances involving singing, playing instruments, dancing, or choreography.",
    "art_creativity": "Creative expression such as drawing, design, crafts, editing, or other artistic work.",
    "technology_gadgets": "Showcase or explanation of electronic devices, gadgets, tech use, or product demos.",
    "education_tips": "Tutorials, guides, explanations, advice, or practical how-to information.",
    "business_motivation": "Themes of entrepreneurship, career advice, money tips, productivity, or motivational messages.",
}

# Keyword hints mapping for simple fallback scoring
# KEYWORD_MAP: Dict[str, List[str]] = {
#     "fashion_beauty": ["makeup", "skincare", "outfit", "ootd", "style", "hair"],
#     "fitness_sports": ["gym", "workout", "yoga", "run", "exercise", "lifting", "squat"],
#     "food_cooking": ["recipe", "cook", "kitchen", "food", "restaurant", "eat", "chef", "street food", "sizzle"],
#     "travel_nature": ["travel", "trip", "beach", "mountain", "hike", "drone", "sunset", "market", "vendor", "stall"],
#     "comedy_memes": ["meme", "skit", "funny", "joke", "lol", "sketch", "prank"],
#     "pets_animals": ["dog", "cat", "pet", "puppy", "kitten", "animal"],
#     "music_dance": ["dance", "song", "singer", "guitar", "beat", "choreo"],
#     "art_creativity": ["draw", "painting", "sketch", "edit", "photograph", "art", "timelapse"],
#     "technology_gadgets": ["unbox", "gadget", "phone", "laptop", "vr", "tech", "benchmark"],
#     "education_tips": ["tutorial", "how to", "tips", "guide", "learn", "explain", "hack"],
#     "business_motivation": ["startup", "entrepreneur", "motivate", "finance", "career", "productivity", "founder"],
#     "lifestyle": ["vlog", "routine", "day in the life", "home", "cafe", "friends", "cozy", "relax"],
# }

# --- REPLACE existing KEYWORD_MAP with this expanded map ---
KEYWORD_MAP: Dict[str, List[str]] = {
    "fashion_beauty": [
        "makeup", "skincare", "outfit", "ootd", "style", "hair",
        "coat", "boots", "sunglasses", "dress", "pose", "look", "mirror"
    ],
    "fitness_sports": ["gym", "workout", "yoga", "run", "exercise", "lifting", "squat"],
    "food_cooking": [
        "recipe", "cook", "kitchen", "food", "restaurant", "eat", "chef",
        "street food", "sizzle", "plating", "dish", "recipe video"
    ],
    "travel_nature": [
        "travel", "trip", "beach", "mountain", "hike", "drone", "sunset",
        "market", "vendor", "stall", "lantern", "tour", "scenic"
    ],
    "comedy_memes": [
        "meme", "skit", "funny", "joke", "lol", "sketch", "prank",
        "spill", "spill(s)", "falls", "fall", "slip", "slips", "fail", "fails",
        "mishap", "reaction", "pratfall"
    ],
    "pets_animals": ["dog", "cat", "pet", "puppy", "kitten", "animal", "puppies", "kittens"],
    "music_dance": ["dance", "song", "singer", "guitar", "beat", "choreo"],
    "art_creativity": ["draw", "painting", "sketch", "edit", "photograph", "art", "timelapse"],
    "technology_gadgets": ["unbox", "gadget", "phone", "laptop", "vr", "tech", "benchmark"],
    "education_tips": ["tutorial", "how to", "tips", "guide", "learn", "explain", "hack"],
    "business_motivation": ["startup", "entrepreneur", "motivate", "finance", "career", "productivity", "founder"],
    "lifestyle": ["vlog", "routine", "day in the life", "home", "cafe", "friends", "cozy", "relax", "scrolling"],
}

# --- APPEND this helper function (place near _keyword_score or before classify_content_type) ---
def apply_heuristic_overrides(summary: str, label_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Apply small heuristic boosts/insertions based on hard tokens.
    - If market/vendor/stall present -> ensure travel_nature is present (boosted).
    - If pet + comedic verb present -> add comedy_memes if missing or boost it.
    - If fashion tokens present -> add/boost fashion_beauty.
    Returns updated label_scores (may include new labels).
    """
    s = (summary or "").lower()
    # convert to dict for easy manipulation
    scores = {lab: float(sc) for lab, sc in label_scores}

    def bump(label: str, amount: float = 0.25):
        scores[label] = max(scores.get(label, 0.0), scores.get(label, 0.0) + amount)

    # market -> travel
    if any(tok in s for tok in ("market", "vendor", "stall", "lantern")):
        # bump travel_nature
        scores["travel_nature"] = max(scores.get("travel_nature", 0.0), scores.get("travel_nature", 0.0) + 0.35)

    # pet + comedic verbs -> comedy
    pet_tokens = ("dog", "cat", "puppy", "kitten", "pet")
    comedy_verbs = ("spill", "spills", "spilled", "fall", "falls", "fell", "slip", "slips", "fail", "fails", "mishap", "pratfall")
    if any(p in s for p in pet_tokens) and any(v in s for v in comedy_verbs):
        scores["comedy_memes"] = max(scores.get("comedy_memes", 0.0), scores.get("comedy_memes", 0.0) + 0.45)

    # fashion cues -> fashion_beauty
    fashion_cues = ("coat", "boots", "sunglasses", "outfit", "pose", "mirror", "dress")
    if any(f in s for f in fashion_cues):
        scores["fashion_beauty"] = max(scores.get("fashion_beauty", 0.0), scores.get("fashion_beauty", 0.0) + 0.35)

    # produce sorted list
    updated = sorted(scores.items(), key=lambda x: -x[1])
    return updated


# Lazy-loaded zero-shot pipeline handle
_zero_shot = None


def _load_zero_shot_pipeline():
    """
    Lazy load the transformers zero-shot-classification pipeline.
    Attempts to put it on GPU if available. Returns None if unavailable.
    """
    global _zero_shot
    if _zero_shot is not None:
        return _zero_shot
    try:
        from transformers import pipeline
        import torch
            # prefer DeBERTa-base-mnli for better NLI on noisy captions
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading zero-shot-classification pipeline on device %s", device)
        _zero_shot = pipeline("zero-shot-classification",  model=model_name, device=device)
        return _zero_shot
    except Exception as e:
        logger.warning("Zero-shot pipeline not available: %s", e)
        _zero_shot = None
        return None


def _keyword_score(summary: str) -> List[Tuple[str, float]]:
    """
    Return a list of (category, score) sorted desc using simple keyword counts.
    Scores are normalized to [0,1] by dividing by the max count (if any).
    """
    s = (summary or "").lower()
    counts: Dict[str, int] = {k: 0 for k in CATEGORY_LIST}
    for cat, kws in KEYWORD_MAP.items():
        for kw in kws:
            if kw in s:
                counts[cat] += 1
    max_count = max(counts.values()) if counts else 0
    if max_count == 0:
        # no signal
        return [(cat, 0.0) for cat in CATEGORY_LIST]
    scored = sorted(((cat, float(counts[cat]) / float(max_count)) for cat in CATEGORY_LIST), key=lambda x: -x[1])
    return scored


def _normalize_and_take_top(items: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
    """
    Sort descending by score and return top_k entries.
    """
    sorted_items = sorted(items, key=lambda x: -x[1])
    return sorted_items[:top_k]


def classify_content_type(
    summary: str,
    top_k: int = 1,
    use_zero_shot: bool = True,
    score_threshold: float = 0.30,
    multi_label: bool = False,
    return_scores: bool = False,
) -> List[Any]:
    """
    Classify a short textual `summary` into category labels.

    Parameters:
      - summary: input text
      - top_k: number of labels to return
      - use_zero_shot: try transformers pipeline if available
      - score_threshold: for multi-label, only return labels with score >= threshold
      - multi_label: if True, enable multi-label scoring (returns top_k labels by score)
      - return_scores: if True, return List[(label, score)], otherwise List[label]

    Returns:
      - If return_scores=True: List[Tuple[label, score]]
      - Else: List[label]
    """
    if not summary:
        return [] if not return_scores else []

    # Try zero-shot pipeline
    if use_zero_shot:
        zsp = _load_zero_shot_pipeline()
        if zsp is not None:
            try:
                candidate_labels = CATEGORY_LIST[:]

                # Call pipeline with multi_label according to argument.
                out = zsp(
                    summary,
                    candidate_labels,
                    hypothesis_template="This example is about {}.",
                    multi_label=multi_label,
                )

                # Normalize output into list of (label, score)
                label_scores: List[Tuple[str, float]] = []
                # apply heuristics


                if isinstance(out, dict):
                    # single example -> dict
                    labels = out.get("labels", [])
                    scores = out.get("scores", [])
                    # labels & scores align
                    for l, s in zip(labels, scores):
                        label_scores.append((l, float(s)))
                elif isinstance(out, list) and out:
                    # batch mode -> list of dicts; take first
                    first = out[0]
                    labels = first.get("labels", [])
                    scores = first.get("scores", [])
                    for l, s in zip(labels, scores):
                        label_scores.append((l, float(s)))
                else:
                    logger.warning("Unexpected zero-shot output format; falling back to keyword.")
                    raise ValueError("unexpected zero-shot output")

                # --- APPLY HEURISTIC OVERRIDES TO ZERO-SHOT SCORES ---
                # label_scores is currently a list like [('label', score), ...]
                label_scores = apply_heuristic_overrides(summary, label_scores)
                # If multi_label False, pipeline returns a single label (labels[0]); convert appropriately
                if not multi_label:
                    if label_scores:
                        best_label, best_score = label_scores[0]
                        if return_scores:
                            return [(best_label, best_score)]
                        return [best_label]
                    # fallthrough to keyword
                else:
                    # Filter by score_threshold
                    filtered = [(lab, sc) for lab, sc in label_scores if sc >= score_threshold]
                    if not filtered:
                        # fallback to keyword if zero-shot didn't find confident labels
                        kw = _keyword_score(summary)
                        kw_filtered = [(k, s) for k, s in kw if s > 0]
                        if not kw_filtered:
                            return [] if not return_scores else []
                        # apply heuristics to keyword scores too
                        kw_filtered = apply_heuristic_overrides(summary, kw_filtered)
                        # return normalized keyword scores
                        top = _normalize_and_take_top(kw_filtered, top_k)
                        return top if return_scores else [k for k, _ in top]
                    top = _normalize_and_take_top(filtered, top_k)
                    return top if return_scores else [k for k, _ in top]

            except Exception as e:
                logger.exception("Zero-shot classification failed: %s", e)
                # fall through to keyword fallback

    # Keyword fallback (or zero-shot unavailable)
    kw_scores = _keyword_score(summary)
    # apply heuristic overrides to keyword scores
    kw_scores = apply_heuristic_overrides(summary, kw_scores)
    # take top_k with their pseudo-scores
    top = _normalize_and_take_top(kw_scores, top_k)
    if not return_scores:
        return [k for k, _ in top]
    return top
