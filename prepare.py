import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
from itertools import groupby
from pathlib import Path
from urllib.request import urlopen

GROUPS = ("1", "2-5", "6-10", "11-50", "50+")
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA_URL = "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/14578232bb6cdf290fea9dd7fee60a3e01ee6fd5/"
ROOTS = {
    "T005",
    "T007",
    "T017",
    "T022",
    "T031",
    "T033",
    "T037",
    "T038",
    "T058",
    "T062",
    "T074",
    "T082",
    "T091",
    "T092",
    "T097",
    "T098",
    "T103",
    "T168",
    "T170",
    "T201",
    "T204",
}
SOURCES = {
    "CPT",
    "FMA",
    "GO",
    "HGNC",
    "HPO",
    "ICD10",
    "ICD10CM",
    "ICD9CM",
    "MDR",
    "MSH",
    "MTH",
    "NCBI",
    "NCI",
    "NDDF",
    "NDFRT",
    "OMIM",
    "RXNORM",
    "SNOMEDCT_US",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def order(rows, seed=0):
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['id']}".encode()).digest(),
            row["id"],
        ),
    )


def model(output):
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(MODEL, revision=MODEL_REVISION).text_config
    net = Qwen3_5ForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        config=config,
        dtype=torch.bfloat16,
        device_map="cpu",
        key_mapping={r"^model\.language_model\.": "model."},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    net.config.eos_token_id = net.generation_config.eos_token_id = (
        tokenizer.eos_token_id
    )
    net.save_pretrained(output, max_shard_size="2GB")
    tokenizer.save_pretrained(output)


def data(output):
    splits = {}
    for split, suffix in [("train", "trng"), ("validation", "dev"), ("test", "test")]:
        pmids = (
            urlopen(DATA_URL + f"full/data/corpus_pubtator_pmids_{suffix}.txt")
            .read()
            .decode()
            .split()
        )
        splits.update({pmid: split for pmid in pmids})
    documents, rows = [], {split: [] for split in ("train", "validation", "test")}
    with urlopen(DATA_URL + "st21pv/data/corpus_pubtator.txt.gz") as response:
        with gzip.open(response, "rt", encoding="utf8") as stream:
            blocks = (
                list(group)
                for blank, group in groupby(stream, lambda line: bool(line.strip()))
                if blank
            )
            for block in blocks:
                pmid, _, title = block[0].rstrip("\r\n").split("|", 2)
                abstract = block[1].rstrip("\r\n").split("|", 2)[2]
                documents.append({"pmid": pmid, "text": title + " " + abstract})
                for i, line in enumerate(block[2:]):
                    _, start, end, surface, types, cui = line.rstrip("\r\n").split("\t")
                    start, end = int(start), int(end)
                    rows[splits[pmid]].append(
                        {
                            "id": f"{pmid}:{start}:{end}:{i}",
                            "pmid": pmid,
                            "start": start,
                            "end": end,
                            "surface": surface,
                            "gold_cuis": [cui.removeprefix("UMLS:")],
                            "semantic_types": types.split(","),
                        }
                    )
    write_jsonl(output / "documents.jsonl", documents)
    for split, examples in rows.items():
        write_jsonl(output / f"{split}.jsonl", examples)


def rrf(path):
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            yield line.rstrip("\r\n").split("|")


def ontology(meta, srdef, output):
    definitions = {row[1]: (row[2], row[3]) for row in rrf(srdef) if row[0] == "STY"}
    trees = [definitions[root][1] for root in ROOTS]
    eligible_types = {
        tui
        for tui, (_, tree) in definitions.items()
        if any(tree == root or tree.startswith(root + ".") for root in trees)
    }
    eligible = {}
    for cui, rows in groupby(rrf(meta / "MRSTY.RRF"), key=lambda row: row[0]):
        types = {row[1] for row in rows}
        if types & eligible_types:
            eligible[cui] = sorted({definitions[tui][0] for tui in types})
    members, names = set(), {}
    for row in rrf(meta / "MRCONSO.RRF"):
        cui, language, term = row[0], row[1], row[14]
        if cui not in eligible:
            continue
        if row[11] in SOURCES:
            members.add(cui)
        priority = (
            language == "ENG",
            row[16] == "N",
            row[2] == "P",
            row[4] == "PF",
            row[6] == "Y",
        )
        previous = names.get(cui)
        if (
            previous is None
            or priority > previous[0]
            or (priority == previous[0] and term < previous[1])
        ):
            names[cui] = (priority, term)
    connection = sqlite3.connect(output / "ontology.sqlite")
    connection.executescript(
        "CREATE TABLE concepts(cui TEXT PRIMARY KEY,label TEXT,types TEXT);"
        "CREATE TABLE terms(id INTEGER PRIMARY KEY,term TEXT NOT NULL UNIQUE);"
        "CREATE TABLE aliases(cui TEXT,term_id INTEGER,PRIMARY KEY(cui,term_id)) WITHOUT ROWID;"
    )
    universe = members & eligible.keys()
    connection.executemany(
        "INSERT INTO concepts VALUES (?,?,?)",
        ((cui, names[cui][1], json.dumps(eligible[cui])) for cui in sorted(universe)),
    )
    batch = []

    def insert():
        connection.executemany(
            "INSERT OR IGNORE INTO terms(term) VALUES (?)",
            ((term,) for term in dict.fromkeys(term for _, term in batch)),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO aliases SELECT ?,id FROM terms WHERE term=?", batch
        )
        connection.commit()
        batch.clear()

    for row in rrf(meta / "MRCONSO.RRF"):
        if row[0] in universe and row[1] == "ENG" and row[16] == "N" and row[14]:
            batch.append((row[0], row[14]))
        if len(batch) >= 10000:
            insert()
    insert()
    connection.execute("CREATE INDEX alias_term ON aliases(term_id,cui)")
    connection.commit()
    connection.close()


def index(ontology, output, device, threads):
    import faiss
    from retrieval import SapBERT

    faiss.omp_set_num_threads(threads)
    encoder = SapBERT(device)
    net = faiss.IndexHNSWFlat(
        encoder.model.config.hidden_size, 32, faiss.METRIC_INNER_PRODUCT
    )
    net.hnsw.efConstruction, net.hnsw.efSearch = 200, 256
    connection = sqlite3.connect(ontology / "ontology.sqlite")
    cursor = connection.execute("SELECT term FROM terms ORDER BY id")
    while batch := cursor.fetchmany(256):
        net.add(encoder.encode([row[0] for row in batch]))
    faiss.write_index(net, str(output / "aliases.faiss"))
    connection.close()
    shutil.copyfile(ontology / "ontology.sqlite", output / "ontology.sqlite")


def cohort(data, index, output, threads):
    from retrieval import Retriever

    retriever = Retriever(index, threads)
    selection, weights = [], {}
    for split in ("train", "validation"):
        rows = read_jsonl(data / f"{split}.jsonl")
        groups = {group: [] for group in GROUPS}
        for row in rows:
            candidates = [
                candidate["cui"] for candidate in retriever.search(row["surface"], 50)
            ]
            rank = next(
                (i for i, cui in enumerate(candidates, 1) if cui in row["gold_cuis"]),
                51,
            )
            group = (
                "1"
                if rank == 1
                else "2-5"
                if rank <= 5
                else "6-10"
                if rank <= 10
                else "11-50"
                if rank <= 50
                else "50+"
            )
            groups[group].append(row)
        count = min(map(len, groups.values())) if split == "train" else 200
        selected = {
            row["id"]: group for group in GROUPS for row in order(groups[group])[:count]
        }
        write_jsonl(
            output / f"{split}.jsonl", [row for row in rows if row["id"] in selected]
        )
        selection.extend(
            {"id": key, "difficulty_group": group} for key, group in selected.items()
        )
        if split == "validation":
            weights = {
                group: {"original_split_weight": len(groups[group]) / len(rows)}
                for group in GROUPS
            }
    write_jsonl(output / "selection.jsonl", selection)
    shutil.copyfile(data / "documents.jsonl", output / "documents.jsonl")
    (output / "manifest.json").write_text(
        json.dumps({"sampling": {"groups": {"validation": weights}}}, indent=2)
    )


def merge(base, adapter, output):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    net = Qwen3_5ForCausalLM.from_pretrained(
        base, dtype=torch.float32, device_map="cpu"
    )
    net = PeftModel.from_pretrained(net, adapter).merge_and_unload().to(torch.bfloat16)
    net.save_pretrained(output, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(adapter).save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["model", "data", "ontology", "index", "cohort", "merge"]
    )
    parser.add_argument("--output", type=Path, required=True)
    for name in ("meta", "srdef", "ontology", "data", "index", "base", "adapter"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in ("model", "data"):
        globals()[args.command](args.output)
    elif args.command == "ontology":
        ontology(args.meta, args.srdef, args.output)
    elif args.command == "index":
        index(args.ontology, args.output, args.device, args.threads)
    elif args.command == "cohort":
        cohort(args.data, args.index, args.output, args.threads)
    else:
        merge(args.base, args.adapter, args.output)
