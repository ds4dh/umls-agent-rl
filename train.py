import argparse
import json
import random
import time
import tomllib
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Qwen3_5ForCausalLM,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_callback import ExportableState
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import add_response_schema, get_training_chat_template
from trl.trainer.utils import RepeatSampler

from agent import Environment, Reward, example
from prepare import GROUPS, order, read_jsonl
from retrieval import Retriever


class Sampler(RepeatSampler):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.size = len(dataset)
        self.padding = (-self.size) % self.batch_size
        self.groups = {group: [] for group in GROUPS}
        for i, row in enumerate(dataset):
            self.groups[row["difficulty_group"]].append(i)
        self.num_samples += self.padding
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        mapping = list(range(self.size))
        rng = random.Random(f"{self.seed}:{self.epoch}:padding")
        mapping.extend(
            rng.choice(self.groups[GROUPS[i % len(GROUPS)]])
            for i in range(self.padding)
        )
        for index in super().__iter__():
            yield mapping[index]


class Trainer(GRPOTrainer):
    def _get_train_sampler(self, dataset=None):
        return Sampler(
            self.train_dataset if dataset is None else dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        **kwargs,
    ):
        if not model.training and self.args.chat_template_kwargs["enable_thinking"]:
            batch_size = 1
        return super()._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            compute_aux_loss=compute_aux_loss,
            **kwargs,
        )

    def evaluate(self, *args, **kwargs):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        backend = self.vllm_generation
        generation_kwargs = dict(backend.generation_kwargs or {})
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                set_seed(0)
                backend.generation_kwargs = {
                    key: value
                    for key, value in generation_kwargs.items()
                    if key != "seed"
                }
                return super().evaluate(*args, **kwargs)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            backend.generation_kwargs = generation_kwargs

    def evaluation_loop(self, *args, **kwargs):
        reward = self.reward_funcs[0]
        reward.correct = {}
        result = super().evaluation_loop(*args, **kwargs)
        counts, correct = Counter(), Counter()
        for mention_id, success in reward.correct.items():
            group = self.membership[mention_id]
            counts[group] += 1
            correct[group] += success
        prefix = kwargs.get("metric_key_prefix", "eval")
        metrics = {
            f"{prefix}_population_accuracy": sum(
                self.weights[g] * correct[g] / counts[g] for g in GROUPS
            )
        }
        metrics[f"{prefix}_balanced_accuracy"] = sum(correct.values()) / sum(
            counts.values()
        )
        metrics.update(
            {f"{prefix}_accuracy_rank_{g}": correct[g] / counts[g] for g in GROUPS}
        )
        return result._replace(metrics={**result.metrics, **metrics})


class Validation(TrainerCallback):
    def __init__(self, updates, checks):
        self.updates = updates
        self.boundaries = {
            (i * updates + checks // 2) // checks for i in range(1, checks + 1)
        }

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step - 1) % self.updates + 1 in self.boundaries:
            control.should_evaluate = control.should_save = True
        return control


class Budget(TrainerCallback, ExportableState):
    def __init__(self, seconds=604800, elapsed=0):
        self.seconds, self.elapsed = seconds, elapsed
        self.start = time.monotonic()

    def state(self):
        return {
            "args": {
                "seconds": self.seconds,
                "elapsed": self.elapsed + time.monotonic() - self.start,
            },
            "attributes": {},
        }

    def on_step_end(self, args, state, control, **kwargs):
        if self.elapsed + time.monotonic() - self.start >= self.seconds:
            control.should_save = control.should_training_stop = True
        return control


class Initial(TrainerCallback):
    def __init__(self, trainer, resume):
        self.trainer, self.resume = trainer, resume

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resume is None:
            metrics = self.trainer.evaluate()
            self.trainer._determine_best_metric(metrics, trial=None)
            state.best_model_checkpoint = str(Path(args.output_dir) / "checkpoint-0")
            self.trainer._save_checkpoint(self.trainer.model, trial=None)
        return control


def arguments(config, output):
    return GRPOConfig(
        output_dir=str(output),
        learning_rate=config["learning_rate"],
        max_steps=-1,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        num_generations=8,
        generation_batch_size=64,
        num_generations_eval=1,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=config["updates_per_block"],
        save_strategy="steps",
        save_steps=config["updates_per_block"],
        save_total_limit=2,
        load_best_model_at_end=True,
        restore_callback_states_from_checkpoint=True,
        metric_for_best_model="population_accuracy",
        greater_is_better=True,
        num_iterations=1,
        loss_type="dr_grpo",
        scale_rewards="none",
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.2,
        importance_sampling_level="token",
        max_completion_length=config["completion_tokens"],
        max_tool_calling_iterations=5,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        chat_template_kwargs={"enable_thinking": config["enable_thinking"]},
        mask_truncated_completions=False,
        disable_dropout=True,
        bf16=True,
        fp16=False,
        cast_lm_head_to_fp32=False,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=config["model_context"],
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="sequence_mask",
        vllm_importance_sampling_clip_min=None,
        vllm_importance_sampling_clip_max=None,
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to="none",
        seed=0,
        data_seed=0,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=False,
    )


def train(data, index, model, output, config, resume=None, threads=4):
    set_seed(0)
    torch.set_num_threads(threads)
    documents = {
        row["pmid"]: row["text"] for row in read_jsonl(data / "documents.jsonl")
    }
    membership = {
        row["id"]: row["difficulty_group"]
        for row in read_jsonl(data / "selection.jsonl")
    }
    datasets = {}
    for split in ("train", "validation"):
        rows = order(read_jsonl(data / f"{split}.jsonl"))
        datasets[split] = Dataset.from_list(
            [
                {
                    **example({**row, "text": documents[row["pmid"]]}),
                    "difficulty_group": membership[row["id"]],
                }
                for row in rows
            ]
        )
    manifest = json.loads((data / "manifest.json").read_text())
    retriever = Retriever(index, threads)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if not getattr(tokenizer, "response_template", None):
        add_response_schema(tokenizer)
    template = get_training_chat_template(processing_class=tokenizer)
    if template is not None:
        tokenizer.chat_template = template
    if getattr(tokenizer, "response_template", None) is not None:
        tokenizer.init_kwargs["response_template"] = tokenizer.response_template
    tokenizer.padding_side = "left"
    backbone = Qwen3_5ForCausalLM.from_pretrained(
        model,
        dtype=torch.float32,
        device_map={"": 0},
        attn_implementation="sdpa",
        use_kernels=True,
    )
    reward = Reward(tokenizer.eos_token_id)
    trainer = Trainer(
        model=backbone,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=64,
            lora_alpha=128,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        ),
        args=arguments(config, output),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        reward_funcs=[reward],
        environment_factory=partial(Environment, retriever),
        callbacks=[
            Validation(config["updates_per_block"], config["validation_checks"]),
            Budget(config["compute_seconds"]),
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"]
            ),
        ],
    )
    trainer.membership = membership
    trainer.weights = {
        group: manifest["sampling"]["groups"]["validation"][group][
            "original_split_weight"
        ]
        for group in GROUPS
    }
    trainer.add_callback(Initial(trainer, resume))
    set_seed(0)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    trainer.save_model(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    (output / "selection.json").write_text(
        json.dumps(
            {
                "checkpoint": trainer.state.best_model_checkpoint,
                "validation_accuracy": trainer.state.best_metric,
                "updates": trainer.state.global_step,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "index", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("thinking.toml"))
    parser.add_argument("--resume")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    train(
        args.data,
        args.index,
        args.model,
        args.output,
        config,
        args.resume,
        args.threads,
    )
