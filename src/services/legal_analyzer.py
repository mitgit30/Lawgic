import logging
import re
from functools import lru_cache

from huggingface_hub import snapshot_download
from transformers import pipeline

from src.core.config import get_settings
from src.models.schemas import ClauseAnalysis
from src.utils.hf_paths import has_hf_weights, resolve_hf_model_path

logger = logging.getLogger(__name__)

LEGAL_LABELS = [
    "payment",
    "penalty",
    "termination",
    "liability",
    "confidentiality",
    "renewal",
    "governing_law",
    "dispute_resolution",
    "intellectual_property",
    "warranty",
]

KEYWORD_MAP = {
    "payment": ["payment", "fees", "invoice", "consideration", "charge"],
    "penalty": ["penalty", "fine", "liquidated damages", "default"],
    "termination": ["termination", "terminate", "breach", "notice period"],
    "liability": ["liability", "indemnify", "damages", "hold harmless", "unlimited"],
    "confidentiality": ["confidential", "non-disclosure", "nda", "proprietary"],
    "renewal": ["renewal", "auto-renew", "automatic renewal", "extend term"],
    "governing_law": ["governing law", "jurisdiction", "india", "arbitration act"],
    "dispute_resolution": ["arbitration", "dispute", "conciliation", "tribunal"],
    "intellectual_property": ["intellectual property", "copyright", "trademark", "patent"],
    "warranty": ["warranty", "representation", "fitness", "merchantability"],
}


def _extract_entities(clause_text: str) -> list[dict]:
    entities = []
    amounts = re.findall(r"(?:INR|Rs\.?)\s?[\d,]+(?:\.\d+)?", clause_text, flags=re.IGNORECASE)
    dates = re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        clause_text,
        flags=re.IGNORECASE,
    )
    for amount in amounts:
        entities.append({"type": "amount", "value": amount})
    for date in dates:
        entities.append({"type": "date", "value": date})
    return entities


class LegalClauseAnalyzer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._classifier = self._load_classifier()

    def _load_classifier(self):
        primary_source = self.settings.legal_model_local_path or self.settings.legal_model_name
        resolved_model_path = resolve_hf_model_path(primary_source)

        if not has_hf_weights(resolved_model_path):
            logger.warning(
                "CustomInLawBERT weights not found at '%s'. Attempting download for '%s'.",
                resolved_model_path,
                self.settings.legal_model_name,
            )
            try:
                downloaded_path = snapshot_download(repo_id=self.settings.legal_model_name)
                resolved_model_path = resolve_hf_model_path(downloaded_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CustomInLawBERT download failed: %s", exc)

        if not has_hf_weights(resolved_model_path):
            logger.warning(
                "CustomInLawBERT has no usable weights at '%s'. Falling back to rule-based classification.",
                resolved_model_path,
            )
            return None

        try:
            logger.info("Loading legal model from: %s", resolved_model_path)
            return pipeline(
                "text-classification",
                model=resolved_model_path,
                tokenizer=resolved_model_path,
                truncation=True,
                max_length=self.settings.max_clause_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load CustomInLawBERT, using rule-based classification: %s", exc)
            return None

    @staticmethod
    def _heuristic_label(clause_text: str) -> tuple[str, float]:
        text = clause_text.lower()
        scores = {}
        for label, words in KEYWORD_MAP.items():
            scores[label] = sum(1 for w in words if w in text)
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        if best_score == 0:
            return "other", 0.3
        confidence = min(0.55 + best_score * 0.1, 0.95)
        return best_label, confidence

    def _predict(self, clause_text: str) -> tuple[str, float]:
        if self._classifier is None:
            return self._heuristic_label(clause_text)

        try:
            prediction = self._classifier(clause_text, top_k=1)[0]
            label = str(prediction.get("label", "other")).lower()
            score = float(prediction.get("score", 0.0))
            if label.startswith("label_"):
                idx = int(label.replace("label_", ""))
                label = LEGAL_LABELS[idx] if idx < len(LEGAL_LABELS) else "other"
            return label, score
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model inference failed, using heuristic classifier: %s", exc)
            return self._heuristic_label(clause_text)

    def analyze_clauses(self, clauses: list[str]) -> list[ClauseAnalysis]:
        if not clauses:
            return []

        analyses: list[ClauseAnalysis] = []
        if self._classifier is not None:
            try:
                predictions = self._classifier(
                    clauses,
                    top_k=1,
                    batch_size=16,
                    truncation=True,
                    max_length=self.settings.max_clause_tokens,
                )
                for idx, clause in enumerate(clauses, start=1):
                    prediction = predictions[idx - 1][0] if isinstance(predictions[idx - 1], list) else predictions[idx - 1]
                    label = str(prediction.get("label", "other")).lower()
                    score = float(prediction.get("score", 0.0))
                    if label.startswith("label_"):
                        label_idx = int(label.replace("label_", ""))
                        label = LEGAL_LABELS[label_idx] if label_idx < len(LEGAL_LABELS) else "other"

                    analyses.append(
                        ClauseAnalysis(
                            clause_id=idx,
                            clause_text=clause,
                            clause_type=label,
                            confidence=score,
                            entities=_extract_entities(clause),
                        )
                    )
                return analyses
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch model inference failed, falling back to per-clause mode: %s", exc)

        for idx, clause in enumerate(clauses, start=1):
            label, confidence = self._predict(clause)
            analyses.append(
                ClauseAnalysis(
                    clause_id=idx,
                    clause_text=clause,
                    clause_type=label,
                    confidence=confidence,
                    entities=_extract_entities(clause),
                )
            )
        return analyses


@lru_cache(maxsize=1)
def get_legal_analyzer() -> LegalClauseAnalyzer:
    return LegalClauseAnalyzer()
