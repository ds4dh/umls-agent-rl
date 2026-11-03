import json
import re

SYSTEM = (
    "Normalize the biomedical mention enclosed by <target> and </target> to a UMLS CUI. "
    "Use the document context to resolve abbreviations, ambiguity, and specificity. "
    "Search UMLS with concise biomedical terms. You may reformulate a query and search again. "
    "Issue one search at a time. Choose a CUI that appears in the search results. "
    "Your final response must contain only that CUI."
)
TOOL_DESCRIPTION = (
    "Search UMLS for distinct biomedical concepts within the episode's fixed budget.\n\n"
    "Args:\n"
    "    query: A concise biomedical name or phrase informed by the document context.\n\n"
    "Returns:\n"
    "    A JSON list of up to 10 distinct concepts, ordered by retrieval score.\n"
    "    Each entry contains cui (concept identifier), label (display name),\n"
    "    types (UMLS semantic types), and alias (the retrieved UMLS name for that concept).\n"
    "    Returns a budget error if the search budget is exhausted."
)
CUI = re.compile(r"C[0-9]{7}", re.ASCII)


def example(row):
    text, start, end = row["text"], row["start"], row["end"]
    return {
        **row,
        "prompt": [
            {"role": "system", "content": SYSTEM + " You have at most 5 searches."},
            {
                "role": "user",
                "content": text[:start]
                + "<target>"
                + text[start:end]
                + "</target>"
                + text[end:],
            },
        ],
    }


class Environment:
    def __init__(self, retriever):
        self.retriever = retriever
        self.calls = 0

    def reset(self, **kwargs):
        self.calls = 0

    def search_umls(self, query: str) -> str:
        if self.calls >= 5:
            return json.dumps(
                {"error": "Search budget exhausted; choose from prior results."}
            )
        self.calls += 1
        return json.dumps(self.retriever.search(query, 10), ensure_ascii=False)


Environment.search_umls.__doc__ = TOOL_DESCRIPTION


def decision(messages):
    if not messages or messages[-1].get("role") != "assistant":
        return None
    final = messages[-1]
    if final.get("tool_calls") or not isinstance(final.get("content"), str):
        return None
    answer = final["content"].strip()
    return answer if CUI.fullmatch(answer) else None


class Reward:
    __name__ = "exact_grounded_cui"

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        self.correct = {}

    def __call__(self, completions, gold_cuis, completion_ids, id, **kwargs):
        rewards = []
        for messages, gold, tokens, mention_id in zip(
            completions, gold_cuis, completion_ids, id
        ):
            answer = decision(messages)
            found = set()
            for message in messages:
                if message.get("role") == "tool":
                    try:
                        entries = json.loads(message["content"])
                    except (ValueError, TypeError):
                        continue
                    if isinstance(entries, list):
                        found.update(
                            entry["cui"]
                            for entry in entries
                            if isinstance(entry, dict) and "cui" in entry
                        )
            self.correct[mention_id] = answer in gold
            rewards.append(
                float(
                    answer in gold
                    and answer in found
                    and bool(tokens)
                    and tokens[-1] == self.eos_token_id
                )
            )
        return rewards
