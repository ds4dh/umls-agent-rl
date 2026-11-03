import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBEDDER = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"


class SapBERT:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDER, revision=REVISION, use_fast=True, do_lower_case=True
        )
        self.tokenizer.padding_side = self.tokenizer.truncation_side = "right"
        self.model = (
            AutoModel.from_pretrained(
                EMBEDDER,
                revision=REVISION,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)

    def encode(self, strings):
        inputs = self.tokenizer(
            strings,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            vectors = self.model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
        return np.ascontiguousarray(vectors, dtype=np.float32)


def candidate_rows(connection, ids, scores, k):
    found = {}
    hits = [(int(i) + 1, float(s)) for i, s in zip(ids, scores) if i >= 0]
    scores_by_term = dict(hits)
    for start in range(0, len(hits), 500):
        batch = hits[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT t.id,a.cui,t.term,c.label,c.types FROM terms t "
            "JOIN aliases a ON a.term_id=t.id JOIN concepts c ON a.cui=c.cui "
            f"WHERE t.id IN ({placeholders})"
        )
        for index, cui, term, label, types in connection.execute(
            query, [i for i, _ in batch]
        ):
            score = scores_by_term[index]
            previous = found.get(cui)
            if (
                previous is None
                or score > previous["_score"]
                or (score == previous["_score"] and term < previous["alias"])
            ):
                found[cui] = {
                    "cui": cui,
                    "label": label,
                    "types": json.loads(types),
                    "alias": term,
                    "_score": score,
                }
    return sorted(found.values(), key=lambda row: (-row["_score"], row["cui"]))[:k]


class Retriever:
    def __init__(self, directory, threads=4):
        directory = Path(directory)
        faiss.omp_set_num_threads(threads)
        self.index = faiss.read_index(str(directory / "aliases.faiss"))
        self.index.hnsw.efSearch = 256
        self.connection = sqlite3.connect(":memory:")
        self.connection.deserialize((directory / "ontology.sqlite").read_bytes())
        self.connection.execute("PRAGMA query_only=ON")
        self.encoder = SapBERT()

    @lru_cache(maxsize=4096)
    def _search(self, query, k):
        vector = self.encoder.encode([query])
        count = min(max(4 * k, 64), self.index.ntotal)
        while count:
            scores, ids = self.index.search(vector, count)
            rows = candidate_rows(self.connection, ids[0], scores[0], k)
            if count == self.index.ntotal or (
                len(rows) == k and float(scores[0][-1]) < rows[-1]["_score"]
            ):
                return json.dumps(
                    [
                        {key: value for key, value in row.items() if key != "_score"}
                        for row in rows
                    ],
                    ensure_ascii=False,
                )
            count = min(2 * count, self.index.ntotal)
        return "[]"

    def search(self, query, k=10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A search needs a nonempty biomedical term")
        return json.loads(self._search(query, k))
