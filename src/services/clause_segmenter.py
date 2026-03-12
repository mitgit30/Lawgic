import re

from src.core.config import get_settings

NUMBERING_PATTERN = re.compile(
    r"(?m)^(?:\d+(?:\.\d+)*[\)\.]|[a-zA-Z][\)\.]|clause\s+\d+(?:\.\d+)*)\s+"
)


def _normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _rough_token_count(text: str) -> int:
    return len(text.split())


def _split_long_clause(clause: str, max_tokens: int) -> list[str]:
    if _rough_token_count(clause) <= max_tokens:
        return [clause]

    sentences = re.split(r"(?<=[\.\?!;])\s+", clause)
    chunks: list[str] = []
    current = []
    current_count = 0

    for sentence in sentences:
        sentence_tokens = _rough_token_count(sentence)
        if sentence_tokens > max_tokens:
            words = sentence.split()
            for start in range(0, len(words), max_tokens):
                chunks.append(" ".join(words[start : start + max_tokens]))
            continue

        if current_count + sentence_tokens > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_count = sentence_tokens
        else:
            current.append(sentence)
            current_count += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def split_into_clauses(text: str) -> list[str]:
    settings = get_settings()
    normalized = _normalize_text(text)
    if not normalized:
        return []

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    clauses: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if NUMBERING_PATTERN.match(paragraph.lower()):
            if buffer:
                clauses.append(buffer.strip())
            buffer = paragraph
        else:
            if buffer:
                buffer = f"{buffer}\n{paragraph}".strip()
            else:
                clauses.append(paragraph)

    if buffer:
        clauses.append(buffer.strip())

    final_chunks: list[str] = []
    for clause in clauses:
        final_chunks.extend(_split_long_clause(clause, settings.max_clause_tokens))

    return final_chunks
