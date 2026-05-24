import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    raw: Dict[str, Any]

    @property
    def model_path(self) -> str:
        return self.raw["model"]["name_or_path"]

    @property
    def train_file(self) -> str:
        return self.raw["data"]["train_file"]

    @property
    def val_file(self) -> str:
        return self.raw["data"]["val_file"]

    @property
    def output_dir(self) -> str:
        return self.raw["training"]["output_dir"]

    @property
    def max_length(self) -> int:
        return int(self.raw["data"]["max_length"])


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path} 第 {idx} 行: {exc}") from exc
    if not samples:
        raise ValueError(f"数据文件为空: {path}")
    return samples


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_messages(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages = sample["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # 兜底模板：当 tokenizer 没有 chat_template 时使用
    parts = []
    for m in messages:
        parts.append(f"{m['role']}: {m['content']}")
    return "\n".join(parts)


def build_dataset(path: str, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    rows = read_jsonl(path)
    texts = [format_messages(item, tokenizer) for item in rows]
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {os.path.basename(path)}",
    )


def build_bnb_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    compute_dtype = q.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_map.get(compute_dtype, torch.bfloat16),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek 7B QLoRA SFT 训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="训练配置文件路径",
    )
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg = TrainConfig(raw_cfg)
    seed = int(raw_cfg["training"].get("seed", 42))
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=raw_cfg["model"].get("trust_remote_code", True),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(raw_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=raw_cfg["model"].get("trust_remote_code", True),
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=raw_cfg["training"].get("gradient_checkpointing", True),
    )

    lora_cfg = raw_cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(cfg.train_file, tokenizer, cfg.max_length)
    eval_ds = build_dataset(cfg.val_file, tokenizer, cfg.max_length)

    training_cfg = raw_cfg["training"]
    precision_cfg = raw_cfg["precision"]
    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_steps=int(training_cfg.get("save_steps", 200)),
        eval_steps=int(training_cfg.get("eval_steps", 200)),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        eval_strategy=training_cfg.get("evaluation_strategy", "steps"),
        report_to=training_cfg.get("report_to", ["tensorboard"]),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        bf16=bool(precision_cfg.get("bf16", True)),
        fp16=bool(precision_cfg.get("fp16", False)),
        seed=seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"训练完成，模型已保存到: {cfg.output_dir}")


if __name__ == "__main__":
    main()
