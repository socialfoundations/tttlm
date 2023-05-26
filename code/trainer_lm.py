"""Fine-tune masked and causal language model on Pile."""

import sys
import os
import argparse
import logging

import numpy as np
import torch

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM

from datasets import load_dataset


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--training_dir', type=str, default='pile/train')
    parser.add_argument('--eval_file', type=str, default='pile/val.jsonl')
    parser.add_argument('--max_steps', type=int, default=8E6)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--model_name', type=str, default='roberta-pile')
    parser.add_argument('--base_model', type=str, default='roberta-large')
    parser.add_argument('--tokenizer', type=str, default='roberta-large')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--masked_lm', action='store_true')
    parser.add_argument('--causal_lm', action='store_true')
    return parser.parse_args()


def train_eval_datasets(tokenizer, pile_training_dir='pile/train',
                                   pile_eval_file='pile/val.jsonl',
                                   masked_lm=True,
                                   mask_probability=0.15,
                                   random_seed=42):
    """Prepare Pile training and eval datasets."""

    pile_training_files = [os.path.join(pile_training_dir, 
                                        str(i).zfill(2)+'.jsonl') for i in range(10)]
    # Set streaming=True to avoid loading the entire dataset into memory 
    train_dataset = load_dataset('json', data_files=pile_training_files,
                                         split='train', streaming=True)
    eval_dataset = load_dataset('json', data_files=pile_eval_file,
                                        split='train', streaming=True)

    torch.manual_seed(random_seed)

    def masked_tokenizer_map(item, mask_probability=mask_probability):
        """Tokenize with padding and truncation, then mask items."""
        result = tokenizer(item['text'], return_tensors='pt',
                           padding='max_length', truncation=True)
        result["labels"] = result['input_ids'].clone()
        rands = torch.rand(result['input_ids'].size())
        result['input_ids'][rands < mask_probability] = tokenizer.mask_token_id
        # ignore loss on not masked tokens
        result['labels'][rands >= mask_probability] = -100
        return result

    def causal_tokenizer_map(item):
        """Tokenize with padding and truncation, then mask items."""
        result = tokenizer(item['text'], return_tensors='pt',
                           padding='max_length', truncation=True)
        result["labels"] = result['input_ids'].clone()
        return result

    if masked_lm:
        tokenizer_map = masked_tokenizer_map
    else:
        tokenizer_map = causal_tokenizer_map

    train_dataset = train_dataset.map(tokenizer_map, batched=True, batch_size=32).remove_columns(['text', 'meta'])
    eval_dataset = eval_dataset.map(tokenizer_map, batched=True, batch_size=32).remove_columns(['text', 'meta'])

    return train_dataset.with_format("torch"), eval_dataset.with_format("torch")


if __name__ == '__main__':

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if args.masked_lm:
        model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    if args.causal_lm:
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

    if not args.masked_lm and not args.causal_lm:
        raise ValueError('Must specify either masked_lm or causal_lm')

    train_dataset, eval_dataset = train_eval_datasets(tokenizer, pile_training_dir=args.training_dir,
                                                                 pile_eval_file=args.eval_file,
                                                                 masked_lm=args.masked_lm)

    output_dir = os.path.join(args.model_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        log_level='warning',
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  eps=args.adam_eps)

    # drop learning rate every 1.5M sequences by factor 10
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1.5E6,
                                                   gamma=0.1)

    trainer = Trainer(
        model=model,
        optimizers = (optimizer, lr_scheduler),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments
    )

    if args.do_train:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            logger.warning(f"Checkpoint detected, resuming training at {last_checkpoint}.")
            trainer.train(resume_from_checkpoint=output_dir)
        else:
            logger.warning("No checkpoint detected, starting training from scratch.")
            trainer.train()

    if args.do_eval:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            eval_results = trainer.evaluate()
            print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}", file=sys.stderr)
            logger.warning(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
        else:
            logger.warning(f"No checkpoint found at {output_dir}.")
