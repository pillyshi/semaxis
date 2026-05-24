from __future__ import annotations

SYSTEM = """\
You are an expert text analyst. Your task is to generate features that distinguish \
one group of texts from another.

Each feature must be defined by:
- hypothesis: a declarative statement about a single text, suitable for NLI scoring \
(e.g. "This text expresses satisfaction with the product." or \
"This text mentions issues with build quality.")

Requirements:
- Each hypothesis must be a statement about a single text, starting with "This text"
- Hypotheses must capture properties that are more characteristic of the positive group \
than the negative group — avoid properties shared equally by both groups
- The hypothesis must be self-contained
- Aim for diverse, non-redundant features

Respond with JSON only:
{"features": [{"hypothesis": "..."}, ...]}
"""

_USER_TEMPLATE = """\
Positive texts (labeled "{pos_label}"):
---
{pos_block}

Negative texts (labeled "{neg_label}"):
---
{neg_block}

Generate exactly {n} features whose hypotheses are more likely to be true for \
"{pos_label}" texts than "{neg_label}" texts.{language_instruction}
"""


def build_user_message(
    pos_texts: list[str],
    neg_texts: list[str],
    pos_label: str,
    neg_label: str,
    n: int,
    language: str | None = None,
) -> str:
    pos_block = "\n---\n".join(pos_texts) if pos_texts else "(none)"
    neg_block = "\n---\n".join(neg_texts) if neg_texts else "(none)"
    language_instruction = (
        f"\nGenerate the hypothesis for each feature in {language}." if language else ""
    )
    return _USER_TEMPLATE.format(
        pos_label=pos_label,
        neg_label=neg_label,
        pos_block=pos_block,
        neg_block=neg_block,
        n=n,
        language_instruction=language_instruction,
    )
