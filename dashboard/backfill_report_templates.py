"""
One-time backfill: converts existing curated bundles' ai_report text from
literal baked-in color names to the new {TEAM_A}/{TEAM_B} placeholder tokens
(see chatbot.substitute_team_tokens and app.py's writing_prompt).

Keys off color_a/color_b, NOT team_a/team_b - confirmed directly against all 3
existing curated bundles that color_a/color_b (set once at CV color-detection
time, never renamed) is what's actually baked into the report text, while
team_a/team_b may have since been renamed away from it (liverpool_psg: report
says "Red"/"Blue", team_a/team_b are now "Liverpool"/"PSG").

Confirmed via direct regex scan of all 3 reports before writing this: only two
literal variants exist per color - the bare word and the possessive ("Red",
"Red's") - no plural form, no lowercase generic-color false positive ("red
card" etc. does not appear). So only those two patterns are handled, both
word-bounded and case-sensitive to avoid touching unrelated text.

Run once: python backfill_report_templates.py
Only ai_report is modified in each bundle.json - every other field, and every
other file (training_plan.json, .cache/), is left untouched.
"""
import json
import re
from pathlib import Path

CURATED_MATCHES_DIR = Path(__file__).parent / "curated_matches"


def backfill_text(text, color_a, color_b):
    count = 0
    for color, token in ((color_a, "{TEAM_A}"), (color_b, "{TEAM_B}")):
        if not color:
            continue
        text, n1 = re.subn(rf"\b{re.escape(color)}'s\b", token + "'s", text)
        text, n2 = re.subn(rf"\b{re.escape(color)}\b", token, text)
        count += n1 + n2
    return text, count


def main():
    for bundle_path in sorted(CURATED_MATCHES_DIR.glob("*/bundle.json")):
        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        color_a, color_b = bundle.get("color_a"), bundle.get("color_b")
        ai_report = bundle.get("ai_report") or ""
        if not ai_report or not color_a or not color_b:
            print(f"SKIP {bundle_path.parent.name}: missing ai_report/color_a/color_b")
            continue
        if "{TEAM_A}" in ai_report or "{TEAM_B}" in ai_report:
            print(f"SKIP {bundle_path.parent.name}: already templated")
            continue

        new_report, count = backfill_text(ai_report, color_a, color_b)
        print(f"{bundle_path.parent.name}: color_a={color_a!r} color_b={color_b!r} -> {count} substitutions")

        bundle["ai_report"] = new_report
        tmp = str(bundle_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        import os
        os.replace(tmp, bundle_path)
        print(f"  wrote {bundle_path}")


if __name__ == "__main__":
    main()
