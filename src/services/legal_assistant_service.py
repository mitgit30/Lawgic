from collections.abc import Iterator

from src.models.schemas import AskRequest, AskResponse, CitationItem, ClauseAnalysis, RetrievedChunk, UploadResponse
from src.services.clause_segmenter import split_into_clauses
from src.services.legal_analyzer import get_legal_analyzer
from src.services.pdf_parser import extract_text_from_pdf
from src.services.rag_service import RAGService
from src.services.risk_engine import detect_risks


class LegalAssistantService:
    def __init__(self) -> None:
        self.analyzer = get_legal_analyzer()
        self._rag_service: RAGService | None = None

    @property
    def rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = RAGService()
        return self._rag_service

    @staticmethod
    def _build_summary(analyses: list[ClauseAnalysis]) -> str:
        if not analyses:
            return "No clauses detected in the uploaded document."

        counts: dict[str, int] = {}
        for item in analyses:
            counts[item.clause_type] = counts.get(item.clause_type, 0) + 1
        top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(f"{label}: {count}" for label, count in top_items)
        return f"Detected {len(analyses)} clauses. Top clause categories: {top_str}."

    def process_document(self, *, file_bytes: bytes, filename: str) -> UploadResponse:
        text = extract_text_from_pdf(file_bytes)
        clauses = split_into_clauses(text)

        if not clauses:
            return UploadResponse(
                filename=filename,
                total_clauses=0,
                summary="No extractable text/clauses found. If this is a scanned PDF, OCR is required.",
                clauses=[],
                risks=[],
            )

        analyses = self.analyzer.analyze_clauses(clauses)
        # Ensure stable contiguous IDs after batch merges.
        for idx, analysis in enumerate(analyses, start=1):
            analysis.clause_id = idx

        risks = detect_risks(analyses)
        summary = self._build_summary(analyses)
        return UploadResponse(
            filename=filename,
            total_clauses=len(analyses),
            summary=summary,
            clauses=analyses,
            risks=risks,
        )

    def answer_question(self, request: AskRequest) -> AskResponse:
        answer, chunks, citations = self.rag_service.answer(
            question=request.question,
            clause_text=request.clause_text,
            top_k=request.top_k,
        )
        return AskResponse(
            question=request.question,
            answer=answer,
            retrieved_context=chunks,
            citations=citations,
        )

    def answer_question_stream(
        self,
        request: AskRequest,
    ) -> tuple[Iterator[str], list[RetrievedChunk], list[CitationItem]]:
        return self.rag_service.answer_stream(
            question=request.question,
            clause_text=request.clause_text,
            top_k=request.top_k,
        )
