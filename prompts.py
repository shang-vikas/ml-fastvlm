# prompts.py
"""
Single-frame human-readable prompt for FastVLM.
Return a concise 1-2 sentence description of what is visible in this frame.
Keep it short, factual, and avoid commentary or speculation.
"""

HUMAN_FRAME_PROMPT = """
Describe this single frame in 1-2 short sentences. Mention the main objects, the primary action (if any), and a brief note about the surroundings/setting.

INPUT:
captions: "{CAPTIONS}"

OUTPUT (human readable):
<1-2 short sentences>
""".strip()
