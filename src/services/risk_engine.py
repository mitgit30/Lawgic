import re

from src.core.config import get_settings
from src.models.schemas import ClauseAnalysis, RiskAssessment


def _extract_inr_amounts(text: str) -> list[float]:
    values = []
    for match in re.findall(r"(?:INR|Rs\.?)\s?([\d,]+(?:\.\d+)?)", text, flags=re.IGNORECASE):
        normalized = match.replace(",", "")
        try:
            values.append(float(normalized))
        except ValueError:
            continue
    return values


def _risk_from_clause(analysis: ClauseAnalysis) -> tuple[str, list[str]]:
    settings = get_settings()
    text = analysis.clause_text.lower()
    triggers: list[str] = []
    risk_score = 0

    if "automatic renewal" in text or "auto-renew" in text:
        triggers.append("automatic_renewal_detected")
        risk_score += 2

    if "unlimited liability" in text or ("liability" in text and "without limit" in text):
        triggers.append("unlimited_liability_detected")
        risk_score += 3

    if analysis.clause_type == "penalty":
        amounts = _extract_inr_amounts(analysis.clause_text)
        if any(amount > settings.penalty_threshold_inr for amount in amounts):
            triggers.append(f"penalty_above_threshold_{int(settings.penalty_threshold_inr)}")
            risk_score += 2

    if analysis.clause_type == "termination" and "without notice" in text:
        triggers.append("termination_without_notice")
        risk_score += 2

    if analysis.clause_type == "governing_law" and "india" not in text:
        triggers.append("non_indian_governing_law")
        risk_score += 2

    if risk_score >= 4:
        return "high", triggers
    if risk_score >= 2:
        return "medium", triggers
    return "low", triggers


def detect_risks(analyses: list[ClauseAnalysis]) -> list[RiskAssessment]:
    assessments: list[RiskAssessment] = []
    for analysis in analyses:
        risk_level, triggers = _risk_from_clause(analysis)
        assessments.append(
            RiskAssessment(
                clause_id=analysis.clause_id,
                clause_text=analysis.clause_text,
                clause_type=analysis.clause_type,
                risk_level=risk_level,
                triggers=triggers,
            )
        )
    return assessments
