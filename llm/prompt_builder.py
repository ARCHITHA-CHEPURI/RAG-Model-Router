
import re


class PromptBuilder:
    FALLBACK = (
        "I'm sorry, but the provided documents do not contain "
        "any information related to your question."
    )

    MAX_CHUNKS = 3
    MAX_CHARS_PER_CHUNK = 3000

    SYSTEM_PROMPT = """
You are a document-grounded question-answering assistant.

Rules:
1. Answer using only information supported by the retrieved context.
2. Never invent facts, examples, statistics, tools, or sources.
3. If the context does not answer the question, return the exact fallback sentence.
4. If the context supports only part of the question, answer only that part.
5. Give a clear, informative answer that directly addresses
   the question. For explanation questions, explain each
   important concept and how the concepts work together.
   Use numbered steps when helpful.
6. Define technical terms accurately. If a definition is
   not supported by the context, do not guess.
7. Do not mention information that is absent from the context.
8. Return only the answer.
9. Do not invent expansions of abbreviations.
10. If a technical definition is not supported by the
    retrieved context, explain only what the context supports.
11. Finish the explanation before stopping. Do not end
    mid-sentence.

Exact fallback sentence:
I'm sorry, but the provided documents do not contain any information related to your question.
""".strip()

    @classmethod
    def build_prompt(cls, query, retrieved_chunks):
        context_parts = []
        retrieved_text = []

        for i, chunk in enumerate(
            retrieved_chunks[:cls.MAX_CHUNKS],
            start=1
        ):
            if not isinstance(chunk, dict):
                continue

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk")
                or ""
            ).strip()

            if not text:
                continue

            text = text[:cls.MAX_CHARS_PER_CHUNK]
            retrieved_text.append(text)

            metadata = chunk.get("metadata") or {}

            filename = (
                chunk.get("source")
                or metadata.get("filename")
                or metadata.get("document_name")
                or "Unknown source"
            )

            chunk_number = metadata.get(
                "chunk_number",
                chunk.get("chunk_number", "Unknown")
            )

            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {filename}\n"
                f"Chunk: {chunk_number}\n"
                f"Content:\n{text}"
            )

        # Do not call the LLM if no usable context exists.
        if not context_parts:
            return None

        if not cls._contains_required_terms(query, "\n".join(retrieved_text)):
            return None

        context = "\n\n".join(context_parts)

        return (
            f"{cls.SYSTEM_PROMPT}\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"USER QUESTION:\n{query.strip()}\n\n"
            "ANSWER:"
        )

    @staticmethod
    def _contains_required_terms(query, context):
        context_terms = " ".join(re.findall(r"\w+", context.casefold()))
        required_terms = re.findall(
            r"\b[A-Z]{3,}\b|\b[A-Za-z]*\d+[A-Za-z0-9]*\b",
            query,
        )
        required_terms.extend(
            re.findall(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}\b",
                query,
            )
        )

        for term in required_terms:
            normalized_term = " ".join(
                re.findall(r"\w+", term.casefold())
            )
            if normalized_term not in context_terms:
                if (
                    term.casefold() == "rrf"
                    and "reciprocal rank fusion" in context_terms
                ):
                    continue
                return False

        return True
