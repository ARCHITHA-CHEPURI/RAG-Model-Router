import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.answer_cache import AnswerCache
from evaluation.router_metrics import analyze_results, load_results
from api.feedback_store import (
    get_model_quality_preference,
    get_feedback_summary,
    record_feedback,
    register_request,
)
from llm.llm_router import LLMRouter
from llm.prompt_builder import PromptBuilder
from llm.router_strategies import BaseRouter, RuleRouter
from retrieval_engine.rag_pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_scored_results_path(configured_path):
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


SCORED_RESULTS_PATH = _resolve_scored_results_path(
    os.getenv(
        "RAG_ROUTER_SCORED_RESULTS",
        "evaluation/benchmark_scored_updated.csv",
    )
)
FEEDBACK_DATABASE_PATH = (
    PROJECT_ROOT / "DATA" / "feedback.sqlite3"
)
ANSWER_CACHE = AnswerCache(max_entries=256, ttl_seconds=300)

app = FastAPI(
    title="Adaptive AI Model Router",
    description=(
        "Benchmark analytics and user feedback for router comparisons."
    ),
    version="0.1.0",
)


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("request_id")
    @classmethod
    def require_non_whitespace(cls, value):
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field must not be blank.")
        return normalized

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value):
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=10000)
    router: Literal["base_router", "rule_router", "ml_router"] = "ml_router"

    @field_validator("query")
    @classmethod
    def require_non_whitespace(cls, value):
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must not be blank.")
        return normalized


@lru_cache(maxsize=1)
def _get_generation_components():
    return (
        RetrievalPipeline(use_cross_encoder=False, min_similarity=0.30),
        LLMRouter(
            feedback_preference_provider=lambda small_model, large_model: (
                get_model_quality_preference(
                    FEEDBACK_DATABASE_PATH,
                    small_model,
                    large_model,
                )
            )
        ),
        BaseRouter(),
        RuleRouter(),
    )


def _load_results():
    if not SCORED_RESULTS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Scored benchmark data is unavailable. Run the benchmark "
                "and scoring steps before requesting analytics."
            ),
        )

    try:
        return load_results(SCORED_RESULTS_PATH)
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load scored benchmark data: {error}",
        ) from error


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/answer")
def generate_answer(request: AnswerRequest):
    request_start = time.perf_counter()
    cached_answer = ANSWER_CACHE.get(request.query, request.router)
    if cached_answer is not None:
        request_id = _register_request(request.router, cached_answer["model"])
        return {
            **cached_answer,
            "request_id": request_id,
            "router": request.router,
            "cache_hit": True,
            "retrieval_seconds": 0.0,
            "generation_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "total_seconds": round(
                time.perf_counter() - request_start,
                4,
            ),
        }

    try:
        retrieval_pipeline, llm_router, base_router, rule_router = (
            _get_generation_components()
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not initialize answer generation: {error}",
        ) from error

    retrieval_start = time.perf_counter()
    try:
        results = retrieval_pipeline.search(request.query, top_k=3)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Document retrieval failed: {error}",
        ) from error
    retrieval_seconds = time.perf_counter() - retrieval_start

    prompt = PromptBuilder.build_prompt(request.query, results)
    generation_seconds = 0.0
    confidence = None
    input_tokens = None
    output_tokens = None
    cost_usd = None

    if prompt is None:
        answer = PromptBuilder.FALLBACK
        model = "none_fallback"
    else:
        strategies = {
            "base_router": base_router,
            "rule_router": rule_router,
            "ml_router": llm_router,
        }
        try:
            model_type, confidence = strategies[request.router].choose_model(
                request.query
            )
            model = llm_router.model_names[model_type]
            generation_start = time.perf_counter()
            model_response = llm_router.clients[
                model_type
            ].generate_response(prompt)
            answer = model_response.text
            input_tokens = model_response.input_tokens
            output_tokens = model_response.output_tokens
            cost_usd = model_response.cost_usd
            generation_seconds = time.perf_counter() - generation_start
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail=f"Answer generation failed: {error}",
            ) from error

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="The selected model returned an empty answer.",
            )
        if answer.startswith("Error:"):
            raise HTTPException(status_code=502, detail=answer)

    sources = list(
        dict.fromkeys(
            (
                result.get("metadata") or {}
            ).get("document_name")
            or (result.get("metadata") or {}).get("filename")
            or result.get("source")
            or "Unknown source"
            for result in results
        )
    )
    answer_data = {
        "answer": answer,
        "model": model,
        "confidence": (
            round(confidence, 3)
            if confidence is not None
            else None
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "sources": sources,
    }
    if prompt is not None:
        ANSWER_CACHE.set(request.query, request.router, answer_data)

    request_id = _register_request(request.router, model)
    return {
        **answer_data,
        "request_id": request_id,
        "router": request.router,
        "cache_hit": False,
        "retrieval_seconds": round(retrieval_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "total_seconds": round(time.perf_counter() - request_start, 4),
    }


def _register_request(router, model):
    request_id = str(uuid4())
    try:
        register_request(
            FEEDBACK_DATABASE_PATH,
            request_id,
            router,
            model,
        )
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not register answer for feedback: {error}",
        ) from error
    return request_id


@app.get("/api/cache/stats")
def cache_stats():
    return ANSWER_CACHE.stats()


@app.get("/api/runs")
def list_runs():
    rows = _load_results()
    runs = sorted(
        {
            (row["run_id"], row["timestamp"])
            for row in rows
            if row.get("run_id")
        },
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    return {"runs": [run_id for run_id, _ in runs]}


@app.get("/api/metrics")
def get_metrics(
    run_id: str | None = Query(
        default=None,
        description="Benchmark run ID; defaults to the newest available run.",
    ),
):
    rows = _load_results()
    try:
        return analyze_results(rows, run_id=run_id)
    except ValueError as error:
        if run_id and not any(row["run_id"] == run_id for row in rows):
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark run {run_id!r} was not found.",
            ) from error
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(feedback: FeedbackSubmission):
    try:
        return record_feedback(FEEDBACK_DATABASE_PATH, feedback)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not persist response feedback: {error}",
        ) from error


@app.get("/api/feedback/summary")
def feedback_summary():
    try:
        return get_feedback_summary(FEEDBACK_DATABASE_PATH)
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read response feedback: {error}",
        ) from error
