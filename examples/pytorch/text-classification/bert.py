#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.integrations import WandbCallback
import torch

from sklearn.metrics import f1_score
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")

os.environ["WANDB_PROJECT"] = "cuneiform_bert"

logger = logging.getLogger(__name__)

class CustomWandbCallback(WandbCallback):
    def __init__(self, trainer=None, test_dataset=None):
        super().__init__()
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.max_val_f1_macro = 0  
        self.max_test_f1_macro = 0  
        self.max_test_f1_micro = 0  
        print("Callback set with test dataset")

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        
        # predictions = self.trainer.predict(self.trainer.train_dataset, metric_key_prefix="predict").predictions
        # predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
        # true_labels = self.trainer.train_dataset['label']
        # f1_macro = f1_score(true_labels, predictions, average='macro')
        # print("f1_macro"*100)
        # print(f1_macro)
        
        train_result = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        print("Training Results - Micro F1:", train_result["eval_micro_f1"], "Macro F1:", train_result["eval_macro_f1"])
        wandb.log({"train_micro_f1": train_result["eval_micro_f1"], "train_macro_f1": train_result["eval_macro_f1"]})

        eval_result = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
        print("Evaluation Results - Micro F1:", eval_result["eval_micro_f1"], "Macro F1:", eval_result["eval_macro_f1"])
        wandb.log({"eval_micro_f1": eval_result["eval_micro_f1"], "eval_macro_f1": eval_result["eval_macro_f1"]})

        test_result = self.trainer.evaluate(eval_dataset=self.test_dataset)
        print("Test Results - Micro F1:", test_result["eval_micro_f1"], "Macro F1:", test_result["eval_macro_f1"])
        wandb.log({"test_micro_f1": test_result["eval_micro_f1"], "test_macro_f1": test_result["eval_macro_f1"]})

        val_f1_macro = eval_result["eval_macro_f1"]
        if val_f1_macro > self.max_val_f1_macro:
            self.max_val_f1_macro = val_f1_macro
            self.max_test_f1_macro = test_result["eval_macro_f1"]
            self.max_test_f1_micro = test_result["eval_micro_f1"]
            wandb.log({"max_val_f1_macro": self.max_val_f1_macro,
                       "max_test_f1_macro": self.max_test_f1_macro,
                       "max_test_f1_micro": self.max_test_f1_micro})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single text."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        # TODO: maybe the larger the better?(but not too large of course. just don't chop inf is enough)
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        return


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_classification", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
     
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    resume_from_checkpoint = True
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}

    # Loading a dataset from local json files
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.
    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    logger.info("Label type is list, doing multi-label classification")
    label_list = [
        "Ur III (ca. 2100-2000 BC)",
        "Old Babylonian (ca. 1900-1600 BC)",
        "Old Akkadian (ca. 2340-2200 BC)",
        "Old Assyrian (ca. 1950-1850 BC)",
        "ED IIIb (ca. 2500-2340 BC)",
        "Early Old Babylonian (ca. 2000-1900 BC)",
        "Neo-Assyrian (ca. 911-612 BC)",
        "ED IIIa (ca. 2600-2500 BC)",
        "Middle Babylonian (ca. 1400-1100 BC)",
        "Middle Assyrian (ca. 1400-1000 BC)",
        "Ebla (ca. 2350-2250 BC)",
        "Lagash II (ca. 2200-2100 BC)",
        "ED I-II (ca. 2900-2700 BC)",
        "Neo-Babylonian (ca. 626-539 BC)"
    ]
    label_list.sort()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # token=model_args.token,
    )

    config.problem_type = "multi_label_classification"
    logger.info("setting problem type to multi label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    # Define special tokens to add
    special_tokens_dict = {'additional_special_tokens': ['<B>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Note: After adding special tokens, if you're using a model alongside,
    # make sure to resize the model's token embeddings to accommodate new tokens

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # token=model_args.token,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    label_to_id = {'ED I-II (ca. 2900-2700 BC)': 0, 'ED IIIa (ca. 2600-2500 BC)': 1, 'ED IIIb (ca. 2500-2340 BC)': 2, 'Early Old Babylonian (ca. 2000-1900 BC)': 3, 'Ebla (ca. 2350-2250 BC)': 4, 'Lagash II (ca. 2200-2100 BC)': 5, 'Middle Assyrian (ca. 1400-1000 BC)': 6, 'Middle Babylonian (ca. 1400-1100 BC)': 7, 'Neo-Assyrian (ca. 911-612 BC)': 8, 'Neo-Babylonian (ca. 626-539 BC)': 9, 'Old Akkadian (ca. 2340-2200 BC)': 10, 'Old Assyrian (ca. 1950-1850 BC)': 11, 'Old Babylonian (ca. 1900-1600 BC)': 12, 'Ur III (ca. 2100-2000 BC)': 13}
    
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            try:
                ids[label_to_id[label]] = 1.0
            except KeyError:
                # continue
                print(f"Warning: Label '{label}' not found in label_to_id dictionary.")
        return ids

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            processed_labels = [labels if isinstance(labels, list) else [labels] for labels in examples["label"]]
            result["label"] = [multi_labels_to_ids(l) for l in processed_labels]
            # result["label"] = [multi_labels_to_ids(examples["label"])]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    predict_dataset = raw_datasets["test"]
    # print(train_dataset["label"])
    # print("1"*1000)
    # Dataset({
    # features: ['sentence', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
    # num_rows: 34896
    # })
    metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.array([np.where(p > 0, 1, 0) for p in preds]) 
        labels = p.label_ids
        # [None, 'micro', 'macro', 'weighted', 'samples']
        micro_f1 = metric.compute(predictions=preds, references=labels, average="micro")
        macro_f1 = metric.compute(predictions=preds, references=labels, average="macro")
        # wandb.log({"eval_micro_f1": micro_f1, "eval_macro_f1": macro_f1})
        return {
            "micro_f1": micro_f1['f1'],
            "macro_f1": macro_f1['f1'],
        }

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    callback = CustomWandbCallback(trainer=trainer, test_dataset=predict_dataset)
    trainer.add_callback(callback) 

    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint 
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_result = trainer.train()

if __name__ == "__main__":
    main()
