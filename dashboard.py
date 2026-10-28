import os

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("RAG_ROUTER_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 180


def request_json(method, api_url, path, *, payload=None, params=None):
    url = f"{api_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method,
            url,
            json=payload,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach router API at {url}: {error}") from error
    except ValueError as error:
        raise RuntimeError(f"Router API returned invalid JSON from {url}.") from error

    if not isinstance(result, dict):
        raise RuntimeError(f"Router API returned an unexpected response from {url}.")
    return result


def _render_accuracy(label, accuracy):
    if accuracy["scored"]:
        st.metric(
            label,
            f"{accuracy['accuracy_percent']:.1f}%",
            help=f"{accuracy['correct']} correct out of {accuracy['scored']}.",
        )
    else:
        st.metric(label, "Not scored")


def _render_benchmark(api_url):
    st.header("Router benchmark")
    runs = request_json("GET", api_url, "/api/runs").get("runs", [])
    if not runs:
        st.info("No benchmark runs are available yet.")
        return

    selected_run = st.selectbox("Benchmark run", runs)
    report = request_json(
        "GET",
        api_url,
        "/api/metrics",
        params={"run_id": selected_run},
    )
    configurations = report["configurations"]
    if not configurations:
        st.info("This benchmark run has no router results.")
        return

    names = list(configurations)
    columns = st.columns(len(names))
    for column, name in zip(columns, names):
        result = configurations[name]
        with column:
            st.subheader(name.replace("_", " ").title())
            st.metric("Queries", result["queries"])
            st.metric("Median total latency", f"{result['median_total_seconds']:.3f} s")
            st.metric("P95 total latency", f"{result['p95_total_seconds']:.3f} s")
            _render_accuracy(
                "Factual heuristic score",
                result["automated_factual_accuracy"],
            )
            _render_accuracy(
                "Out-of-scope heuristic score",
                result["automated_out_of_scope_accuracy"],
            )
            _render_accuracy(
                "Expected source retrieved@3",
                result["expected_source_hit_rate_at_3"],
            )
            manual = result["manual_factual_accuracy"]
            _render_accuracy("Human-reviewed factual score", manual)
            _render_accuracy(
                "Human-reviewed out-of-scope score",
                result["manual_out_of_scope_accuracy"],
            )
            st.metric(
                "Small-model share",
                f"{result['small_model_share_percent']:.1f}%",
            )
            provider_cost = result["litellm_estimated_cost_usd"]
            st.metric(
                "LiteLLM cost estimate",
                (
                    f"${provider_cost:.6f}"
                    if provider_cost is not None
                    else "Not reported"
                ),
                help=(
                    "Estimated API cost based on LiteLLM model pricing; "
                    "verify charges with the provider. "
                    f"Coverage: {result['cost_estimate_coverage_percent']:.1f}% "
                    "of model calls."
                ),
            )

    chart_data = pd.DataFrame.from_dict(
        {
            name: {
                "Median total latency (s)": result["median_total_seconds"],
                "P95 total latency (s)": result["p95_total_seconds"],
            }
            for name, result in configurations.items()
        },
        orient="index",
    )
    st.subheader("Latency by router")
    st.bar_chart(chart_data)

    comparisons = report.get("comparisons_vs_baseline", {})
    if comparisons:
        comparison_rows = []
        for name, values in comparisons.items():
            comparison_rows.append(
                {
                    "Router": name,
                    **values,
                }
            )
        st.subheader(
            f"Compared with {report['comparison_baseline'].replace('_', ' ')}"
        )
        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Router"),
            use_container_width=True,
        )
        quality_rows = [
            {
                "Router": name,
                **values["paired_manual_factual_quality"],
            }
            for name, values in comparisons.items()
        ]
        st.subheader("Human-reviewed quality vs baseline")
        st.dataframe(
            pd.DataFrame(quality_rows).set_index("Router"),
            use_container_width=True,
        )
    st.caption(report["scoring_note"])
    st.caption(report["cost_note"])


def _render_cache(api_url):
    st.header("Answer cache")
    summary = request_json("GET", api_url, "/api/cache/stats")
    first, second, third, fourth = st.columns(4)
    first.metric("Cache hits", summary["hits"])
    second.metric("Cache misses", summary["misses"])
    third.metric("Hit rate", f"{summary['hit_rate_percent']:.1f}%")
    fourth.metric("Entries", f"{summary['entries']} / {summary['max_entries']}")
    st.caption(
        f"TTL: {summary['ttl_seconds']} seconds. "
        f"{summary['scope']}."
    )


def _render_feedback(api_url):
    st.header("Customer feedback")
    summary = request_json("GET", api_url, "/api/feedback/summary")
    left, middle, right = st.columns(3)
    left.metric("Answers issued", summary["requests_issued"])
    middle.metric("Ratings received", summary["responses"])
    rating = summary["average_rating"]
    right.metric(
        "Average rating",
        f"{rating:.2f} / 5" if rating is not None else "No ratings",
    )
    positive_rate = summary["positive_rate_percent"]
    if positive_rate is not None:
        st.metric(
            "Positive ratings",
            f"{positive_rate:.1f}%",
            help=summary["positive_rating_definition"],
        )
    if summary["by_router_and_model"]:
        st.dataframe(
            pd.DataFrame(summary["by_router_and_model"]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(summary["learning_note"])


def _render_playground(api_url):
    st.header("Try a query")
    with st.form("answer-form"):
        query = st.text_area(
            "Question",
            placeholder="Ask a question supported by your documents",
            max_chars=10000,
        )
        router = st.selectbox(
            "Routing strategy",
            ["ml_router", "rule_router", "base_router"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        submitted = st.form_submit_button("Generate answer")

    if submitted:
        if not query.strip():
            st.warning("Enter a question before generating an answer.")
        else:
            try:
                answer = request_json(
                    "POST",
                    api_url,
                    "/api/answer",
                    payload={"query": query, "router": router},
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.session_state["last_answer"] = answer
                st.session_state.pop("feedback_submitted", None)

    answer = st.session_state.get("last_answer")
    if not answer:
        return

    st.subheader("Answer")
    st.write(answer["answer"])
    st.caption(
        f"Router: {answer['router']} | Model: {answer['model']} | "
        f"Total: {answer['total_seconds']:.3f} s | "
        f"Cache: {'hit' if answer['cache_hit'] else 'miss'}"
    )
    if answer["cache_hit"]:
        st.caption("Latency was measured for this cache-hit response.")
    else:
        cost = answer.get("cost_usd")
        cost_text = f"${cost:.6f}" if cost is not None else "not reported"
        st.caption(
            f"Tokens: {answer.get('input_tokens') or 0} input / "
            f"{answer.get('output_tokens') or 0} output | "
            f"Provider-reported cost: {cost_text}"
        )
    if answer["sources"]:
        st.caption("Sources: " + ", ".join(answer["sources"]))

    if st.session_state.get("feedback_submitted") == answer["request_id"]:
        st.success("Thanks — your rating was recorded.")
        return

    with st.form("answer-feedback-form"):
        rating = st.slider("Rate this answer", min_value=1, max_value=5, value=4)
        comment = st.text_input(
            "Optional feedback",
            max_chars=2000,
            placeholder="What was helpful or missing?",
        )
        feedback_submitted = st.form_submit_button("Submit rating")

    if feedback_submitted:
        try:
            request_json(
                "POST",
                api_url,
                "/api/feedback",
                payload={
                    "request_id": answer["request_id"],
                    "rating": rating,
                    "comment": comment or None,
                },
            )
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.session_state["feedback_submitted"] = answer["request_id"]
            st.rerun()


def main():
    st.set_page_config(
        page_title="Adaptive AI Model Router",
        page_icon="⚡",
        layout="wide",
    )
    st.title("Adaptive AI Model Router")
    st.caption("Compare routing performance and collect answer ratings.")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("Router API URL", value=DEFAULT_API_URL)
        st.caption("Start the FastAPI service before loading dashboard data.")

    if st.button("Refresh dashboard"):
        st.rerun()

    try:
        request_json("GET", api_url, "/health")
    except RuntimeError as error:
        st.error(str(error))
        st.info("Start the API with `uvicorn api.app:app --reload`.")
        return

    tabs = st.tabs(["Overview", "Try it"])
    with tabs[0]:
        for render_section in (_render_benchmark, _render_feedback, _render_cache):
            try:
                render_section(api_url)
            except RuntimeError as error:
                st.error(str(error))
    with tabs[1]:
        _render_playground(api_url)


if __name__ == "__main__":
    main()
