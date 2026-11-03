# UMLS agent

Fine-tune Qwen3.5-4B with LoRA and Dr. GRPO for biomedical concept normalisation using a fixed SapBERT–HNSW search tool.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Prepare

Obtain UMLS 2017AA and prepare its Active subset with MetamorphoSys.

```bash
python prepare.py model --output base
python prepare.py data --output data
python prepare.py ontology --meta /path/to/2017AA-Active/META --srdef /path/to/2017AA/NET/SRDEF --output ontology
python prepare.py index --ontology ontology --output index --device cuda --threads 16
python prepare.py cohort --data data --index index --output cohort
```

## Train

```bash
python train.py --data cohort --index index --model base --output runs/thinking --config thinking.toml
```