from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SafetyCategory(str, Enum):
    CYBER = "cybersecurity"
    BIO = "biology_chemistry"
    DISTILL = "distillation"
    CLEAN = "clean"


AccessTier = Literal["public", "glasswing", "bio_trust"]


@dataclass
class RouteDecision:
    category: SafetyCategory
    use_fallback: bool
    model_tier: str
    reason: str


CYBER_PATTERNS = [
    r"exploit",
    r"shellcode",
    r"buffer overflow",
    r"privilege escalation",
    r"lateral movement",
    r"ransomware",
]

BIO_PATTERNS = [
    r"bioweapon",
    r"pathogen synthesis",
    r"toxin production",
    r"gain.of.function",
]

DISTILL_PATTERNS = [
    r"distill",
    r"extract weights",
    r"clone your model",
    r"replicate claude",
]


def classify_query(text: str) -> SafetyCategory:
    lower = text.lower()
    for pat in CYBER_PATTERNS:
        if re.search(pat, lower):
            return SafetyCategory.CYBER
    for pat in BIO_PATTERNS:
        if re.search(pat, lower):
            return SafetyCategory.BIO
    for pat in DISTILL_PATTERNS:
        if re.search(pat, lower):
            return SafetyCategory.DISTILL
    return SafetyCategory.CLEAN


def route_request(
    text: str,
    access_tier: AccessTier = "public",
    fallback_model: str = "mythos-medium",
    primary_model: str = "mythos-frontier",
) -> RouteDecision:
    """
    Mythos/Fable dual-tier routing.

    - public (Fable): cyber/bio/distill → fallback
    - glasswing: cyber safeguards lifted
    - bio_trust: bio safeguards lifted, cyber kept
    """
    category = classify_query(text)

    if access_tier == "glasswing":
        if category == SafetyCategory.DISTILL:
            return RouteDecision(category, True, fallback_model, "distillation blocked")
        return RouteDecision(category, False, primary_model, "glasswing cyber access")

    if access_tier == "bio_trust":
        if category == SafetyCategory.CYBER:
            return RouteDecision(category, True, fallback_model, "cyber blocked for bio tier")
        if category == SafetyCategory.BIO:
            return RouteDecision(category, False, primary_model, "bio trust access")
        if category == SafetyCategory.DISTILL:
            return RouteDecision(category, True, fallback_model, "distillation blocked")
        return RouteDecision(category, False, primary_model, "bio trust default")

    # public Fable tier
    if category != SafetyCategory.CLEAN:
        return RouteDecision(category, True, fallback_model, f"fable safeguard: {category.value}")
    return RouteDecision(category, False, primary_model, "clean query")
