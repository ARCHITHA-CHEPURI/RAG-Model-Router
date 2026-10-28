import sqlite3
from pathlib import Path


def _connect(database_path, create_schema=True):
    path = Path(database_path)
    if not create_schema and not path.is_file():
        return None
    if create_schema:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    if create_schema:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_requests (
                request_id TEXT PRIMARY KEY,
                router TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS response_feedback (
                request_id TEXT PRIMARY KEY,
                router TEXT NOT NULL,
                model TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )
    return connection


def register_request(database_path, request_id, router, model):
    connection = _connect(database_path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO inference_requests (request_id, router, model)
                VALUES (?, ?, ?)
                """,
                (request_id, router, model),
            )
    finally:
        connection.close()


def record_feedback(database_path, feedback):
    connection = _connect(database_path)
    try:
        with connection:
            request = connection.execute(
                """
                SELECT router, model
                FROM inference_requests
                WHERE request_id = ?
                """,
                (feedback.request_id,),
            ).fetchone()
            if request is None:
                raise LookupError(
                    f"Request {feedback.request_id!r} was not issued by this API."
                )
            connection.execute(
                """
                INSERT INTO response_feedback (
                    request_id, router, model, rating, comment
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback.request_id,
                    request["router"],
                    request["model"],
                    feedback.rating,
                    feedback.comment,
                ),
            )
    except sqlite3.IntegrityError as error:
        if "response_feedback.request_id" in str(error):
            raise ValueError(
                f"Feedback for request {feedback.request_id!r} already exists."
            ) from error
        raise
    finally:
        connection.close()

    return {
        "request_id": feedback.request_id,
        "router": request["router"],
        "model": request["model"],
        "rating": feedback.rating,
        "created": True,
    }


def _empty_summary():
    return {
        "responses": 0,
        "average_rating": None,
        "positive_rate_percent": None,
        "positive_rating_definition": "rating >= 4 out of 5",
        "by_router_and_model": [],
        "requests_issued": 0,
        "learning_note": (
            "For low-confidence ML routing decisions, ratings can influence "
            "the selected model after each configured model has at least five "
            "ratings and their average ratings differ by at least 0.5 points. "
            "Only ML-router ratings are counted; otherwise the router keeps "
            "its conservative large-model fallback."
        ),
    }


def get_feedback_summary(database_path):
    connection = _connect(database_path, create_schema=False)
    if connection is None:
        return _empty_summary()

    try:
        overall = connection.execute(
            """
            SELECT
                COUNT(*) AS responses,
                AVG(rating) AS average_rating,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS positive
            FROM response_feedback
            """
        ).fetchone()
        groups = connection.execute(
            """
            SELECT
                router,
                model,
                COUNT(*) AS responses,
                AVG(rating) AS average_rating,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS positive
            FROM response_feedback
            GROUP BY router, model
            ORDER BY router, model
            """
        ).fetchall()
        requests_issued = connection.execute(
            "SELECT COUNT(*) FROM inference_requests"
        ).fetchone()[0]
    finally:
        connection.close()

    responses = overall["responses"]
    summary = _empty_summary()
    summary.update(
        {
            "responses": responses,
            "average_rating": (
                round(overall["average_rating"], 2)
                if overall["average_rating"] is not None
                else None
            ),
            "positive_rate_percent": (
                round(overall["positive"] / responses * 100, 1)
                if responses
                else None
            ),
            "by_router_and_model": [
                {
                    "router": row["router"],
                    "model": row["model"],
                    "responses": row["responses"],
                    "average_rating": round(row["average_rating"], 2),
                    "positive_rate_percent": round(
                        row["positive"] / row["responses"] * 100,
                        1,
                    ),
                }
                for row in groups
            ],
            "requests_issued": requests_issued,
        }
    )
    return summary


def get_model_quality_preference(
    database_path,
    small_model,
    large_model,
    min_responses=5,
    min_rating_gap=0.5,
):
    if small_model == large_model:
        return None

    connection = _connect(database_path, create_schema=False)
    if connection is None:
        return None

    try:
        rows = connection.execute(
            """
            SELECT model, COUNT(*) AS responses, AVG(rating) AS average_rating
            FROM response_feedback
            WHERE router = 'ml_router' AND model IN (?, ?)
            GROUP BY model
            """,
            (small_model, large_model),
        ).fetchall()
    finally:
        connection.close()

    model_stats = {
        row["model"]: {
            "responses": row["responses"],
            "average_rating": row["average_rating"],
        }
        for row in rows
    }
    small_stats = model_stats.get(small_model)
    large_stats = model_stats.get(large_model)
    if (
        small_stats is None
        or large_stats is None
        or small_stats["responses"] < min_responses
        or large_stats["responses"] < min_responses
    ):
        return None

    rating_gap = (
        small_stats["average_rating"] - large_stats["average_rating"]
    )
    if abs(rating_gap) < min_rating_gap:
        return None

    return "small" if rating_gap > 0 else "large"
