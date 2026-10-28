import re


class BaseRouter:
    """Always select the large model as the comparison baseline."""

    def choose_model(self, query):
        self._validate_query(query)
        return "large", None

    @staticmethod
    def _validate_query(query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")


class RuleRouter(BaseRouter):
    """Use transparent complexity rules to choose between local models."""

    COMPLEXITY_TERMS = {
        "analyze",
        "architect",
        "build",
        "complex",
        "complexity",
        "compare",
        "create",
        "debug",
        "design",
        "develop",
        "generate",
        "implement",
        "optimize",
        "tradeoff",
        "tradeoffs",
        "write",
    }

    def choose_model(self, query):
        self._validate_query(query)
        words = re.findall(r"\b\w+\b", query.casefold())
        has_complexity_term = bool(self.COMPLEXITY_TERMS.intersection(words))
        is_long_query = len(words) >= 35
        model = "large" if has_complexity_term or is_long_query else "small"

        return model, None
