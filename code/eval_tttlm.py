"""Evaluate TTT-NN on Pile tasks."""

import os
import time
import logging
import argparse
import random

from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass

import torch
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForCausalLM

from pile_client import roberta_client

from lm_eval.gpt2 import HFLM
from lm_eval.tasks import get_task, ALL_TASKS
from utils import aggregate


def parse_args():
    """Parse command line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--address_path', type=str, default='servers/addresses.txt')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--tokenizer', type=str, default='EleutherAI/gpt-neo-1.3B')
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B')
    parser.add_argument('--embedding_model_checkpoint', type=str,
                        default='models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000')
    parser.add_argument('--tasks', type=str, default='pile_all')
    parser.add_argument('--val_portion', type=float, default=0.2)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--learning_rate', type=float,default=5e-6)
    parser.add_argument('--num_neighbors', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=2048)
    parser.add_argument('--reset_weights', type=bool, default=True)
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--mask_probability', type=float, default=0.15)
    parser.add_argument('--distance_threshold', type=float, default=4.0) # 4.0 filters essentially nothing
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--dynamic_eval', action='store_true')
    parser.add_argument('--split_text', action='store_true')
    return parser.parse_args()


@dataclass
class TTTLMConfig:
    """TTTLM configuration."""
    model : str ='EleutherAI/gpt-neo-1.3B'
    tokenizer : str ='EleutherAI/gpt-neo-1.3B'
    num_neighbors : int = 50
    reset_weights : bool = True
    max_length : int = 2048
    stride : int = 2048
    mask_probability : float = 0.15
    learning_rate : float = 5e-6
    adam_epsilon : float = 1e-8
    distance_threshold : float = 4.0
    dynamic_eval : bool = False
    split_text : bool = False


class TTTLM:
    """Test-time training on nearest neighbors for language models."""

    def __init__(self, pile_client, model, tokenizer, task, 
                 num_fewshot=0, device=torch.device('cuda:0'), config=TTTLMConfig()):
        self.pile_client = pile_client
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr = config.learning_rate, 
                                           eps = config.adam_epsilon)
        self.task = task
        self.num_fewshot = num_fewshot
        self.rnd = random.Random()
        self.eval_model = HFLM(model, tokenizer, device)
        self._original_state = deepcopy(model.state_dict())

    def _evaluate(self, eval_doc):
        """Evaluate using calls to lm_eval."""
        self.eval_model.gpt2.eval()

        if isinstance(eval_doc, str):
            doc, ctx = eval_doc, ''
        else:
            doc, ctx = eval_doc

        reqs = self.task.construct_requests(doc, ctx)
        if not isinstance(reqs, (list, tuple)):
            reqs = [reqs]

        resps = []
        for req in reqs:
            reqtype = req.request_type
            resp = getattr(self.eval_model, reqtype)([req.args])
            resps.append(resp[0])

        resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]
        metrics = self.task.process_results(doc, resps)
        return metrics

    def _reset_model(self):
        """Reset model parameters to original values."""
        self.model.load_state_dict(self._original_state)

    def _input_target_pairs_masked(self, input_ids):
        """Create input-target pairs for training masked LM.
        
        Parameters
        ----------
        input_ids: torch tensor of shape (batch_size, seq_len)
            Input ids from tokenizer
        
        Returns 
        -------
            List of (torch.Tensor, torch.Tensor) pairs."""

        seq_len = input_ids.size(1)
        input_target_pairs = []
        for begin_loc in range(0, seq_len, self.config.stride):
            end_loc = min(begin_loc + self.config.max_length, seq_len)
            inputs = input_ids[:, begin_loc:end_loc]
            targets = inputs.clone()
            rands = np.random.rand(inputs.size(1)) 
            inputs[:, rands < self.config.mask_probability] = self.tokenizer.mask_token_id
            # Ignore loss on unmasked tokens
            targets[:, rands >= self.config.mask_probability] = -100
            input_target_pairs.append((inputs, targets))

        return input_target_pairs

    def _input_target_pairs_causal(self, input_ids):
        """Create input-target pairs for training causal LM.
        
        Parameters
        ----------
        input_ids: torch tensor of shape (batch_size, seq_len)
            Input ids from tokenizer
        
        Returns 
        -------
            List of (torch.Tensor, torch.Tensor) pairs."""

        seq_len = input_ids.size(1)
        input_target_pairs = []
        # This determines how many examples we create from a single sequence
        # Currently, this could be too many for long sequences, and too few for short sequences
        for begin_loc in range(0, seq_len, self.config.stride):
            end_loc = min(begin_loc + self.config.max_length, seq_len)
            inputs = input_ids[:, begin_loc:end_loc]
            targets = inputs.clone()
            input_target_pairs.append((inputs, targets))

        return input_target_pairs

    def _split_text(self, text, max_prefix_length=1024):
        """Split text into query and eval text."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= 2 * max_prefix_length:
            split = max_prefix_length
        else:
            split = len(tokens) // 2
        
        query_text = self.tokenizer.decode(tokens[:split])
        eval_text = self.tokenizer.decode(tokens[split:])
        return query_text, eval_text

    def _filter_retrieved(self, query_text, vectors, texts):
        """Filter retrieved texts based on distance to query embedding."""
        query_vector = self.pile_client.embedding_model([query_text]).cpu().numpy()
        near = np.linalg.norm(vectors - query_vector, axis=1) < self.config.distance_threshold
        # select texts where near is true
        texts = [text for text, n in zip(texts, near) if n]
        return texts

    def _retrieve(self, query_text):
        """Retrieve nearest neighbors given query text.
        
        Parameters
        ----------
        query_text: str
            Query text.
        
        Returns
        -------
        texts: list of str
            Retrieved texts.
        """

        if self.config.dynamic_eval:
            return [query_text] * self.config.num_neighbors
        else:
            # On long runs, retrieval sometimes fails due to network issues.
            try:
                vectors, texts = self.pile_client.string_query(query_text,
                                                        self.config.num_neighbors)
            except Exception as e:
                logging.warning("Failed to retrieve: %s", e)
                logging.warning("Query text: %s", query_text[:1000])
                return []

        return self._filter_retrieved(query_text, vectors, texts)

    def train_single(self, text):
        """Train model on a single text.
        
        Parameters
        ----------
        text: str
            Text to train model on.

        Returns
        -------
        loss: float
            Training loss.
        """
        encodings = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)

        if self.tokenizer.mask_token_id:
            input_target_pairs = self._input_target_pairs_masked(encodings.input_ids)
        else:
            input_target_pairs = self._input_target_pairs_causal(encodings.input_ids)

        self.model.train()
        tr_loss = 0
        
        if len(input_target_pairs) == 0:
            return 0

        for inputs, targets in input_target_pairs:
            self.model.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs, labels=targets)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tr_loss += outputs.loss.item()
        
        return tr_loss / len(input_target_pairs)

    def train(self, doc):
        """Train on k nearest neighbors and test on input text.
        
        Parameters
        ----------
        doc: str or dict
            String when evaluating for perplexity,
            otherwise a dictionary, e.g. in QA, the question and choices.

        Returns
        -------
            Perplexity score before and after training,
            time to retrieve and time to train.
        """
        ctx = self.task.fewshot_context(
                doc=doc, num_fewshot=self.num_fewshot, rnd=self.rnd, description=None
            )

        if isinstance(doc, str):
            # Evaluating with perplexity
            assert(ctx == '')
            if self.config.split_text:
                # Make sure that query and eval text are non overlapping
                query_text, eval_doc = self._split_text(doc)
            else:
                query_text, eval_doc = doc, doc
        else:
            # Evaluating with other metrics
            assert(ctx != '')
            query_text = ctx
            eval_doc = (doc, ctx)

        tstart_retrieve = time.time()
        texts = self._retrieve(query_text)
        c_retr = time.time() - tstart_retrieve
        logging.info("Retrieval time: {:.2f} seconds".format(c_retr))

        c_train = []
        tr_losses = []
        te_stats = [self._evaluate(eval_doc)]

        for text in texts:
            tstart_train = time.time()
            tr_loss = self.train_single(text)
            logging.info("Training loss: {:.2f}".format(tr_loss))
            tr_losses.append(tr_loss)
            te_stats.append(self._evaluate(eval_doc))
            c_train.append(time.time() - tstart_train)
            logging.info("Training time: {:.2f} seconds".format(c_train[-1]))
            
        # Reset weights
        if self.config.reset_weights:
            self._reset_model()

        return te_stats, tr_losses, c_train, c_retr

    def generate(self, prompt):
        """Generate text from prompt before and after training.
        
        Parameters
        ----------
        prompt: str
            Prompt to generate text from.
        
        Returns
        -------
            List of generated texts before and after training.
        """
        encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)

        outputs_before = self.model.generate(input_ids, 
                                             attention_mask=attention_mask, 
                                             max_length=100, 
                                             do_sample=True)
        decoded_before = tokenizer.batch_decode(outputs_before, skip_special_tokens=True)
        logging.info('Before training:\n %s', decoded_before)

        _, texts = self.pile_client.string_query(prompt, self.config.num_neighbors)
        for text in texts:
            self.update(text)

        outputs = self.model.generate(input_ids, 
                                      attention_mask=attention_mask, 
                                      max_length=100, 
                                      do_sample=True) 
        decoded_after = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logging.info('After training:\n %s', decoded_after)

        # Reset weights
        if self.config.reset_weights:
            self._reset_model()

        return decoded_before, decoded_after


def eval_tttlm(tttlm, rank, world_size, results_path, val_portion=1.0):
    """Evaluate TTLM on test set.
    
    Parameters:
    -----------
    tttlm: TTTLM
        TTTLM model to evaluate.
    val_portion: float
        Fraction of validation examples to use. If -1, use all validation examples.
    """

    task = tttlm.task
    if task.has_test_docs():
        task_doc_func = task.test_docs
        task_set = "test"  # Required for caching in the decontamination
        logging.info('Using test docs')
    elif task.has_validation_docs():
        task_set = "val"  # Required for caching in the decontamination
        task_doc_func = task.validation_docs
        logging.info('Using validation docs')
    else:
        raise RuntimeError("Task has neither test_docs nor validation_docs")

    task_docs = list(task_doc_func())
    rnd = random.Random()
    rnd.seed(1216)
    rnd.shuffle(task_docs)

    if val_portion < 1:
        task_docs = task_docs[:int(val_portion * len(task_docs))]

    if world_size > 1:
        my_slice = np.arange(rank, len(task_docs), world_size)
        task_docs = [task_docs[i] for i in my_slice]

    all_stats = []
    all_losses = []
    training_costs = []
    retrieval_costs = []

    before_stats = []
    after_stats = []

    for doc in tqdm(task_docs):
        stats, losses, c_train, c_retr = tttlm.train(doc)

        before_stats.append(stats[0])
        after_stats.append(stats[-1])

        try:
            logging.info('Aggeregate before: %s', aggregate(before_stats, task))
            logging.info('Aggeregate after: %s', aggregate(after_stats, task))
        except:
            pass

        all_stats.append(stats)
        all_losses.append(losses)
        training_costs.append(c_train)
        retrieval_costs.append(c_retr)

        results = (all_stats, all_losses, training_costs, retrieval_costs)
        torch.save(results, results_path)


if __name__ == '__main__':


    args = parse_args()

    config = TTTLMConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        num_neighbors=args.num_neighbors,
        reset_weights=args.reset_weights,
        max_length=args.max_length,
        stride=args.stride,
        mask_probability=args.mask_probability,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        distance_threshold=args.distance_threshold,
        dynamic_eval=args.dynamic_eval,
        split_text=args.split_text,
    )

    logging.getLogger().setLevel(args.logging_level)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logging.info("Loading model: %s", args.model)
    if tokenizer.mask_token_id:
        model = AutoModelForMaskedLM.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.tasks == 'pile_all':
        from utils import PILE_WEIGHTS
        task_names = PILE_WEIGHTS.keys()
    elif args.tasks == 'super_glue':
        task_names = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    else:
        from utils import pattern_match
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    if args.dynamic_eval:
        pile_client = None
    else:
        pile_client = roberta_client(address_path=args.address_path,
                                     embedding_model_checkpoint=args.embedding_model_checkpoint)

    os.makedirs(args.results_dir, exist_ok=True)
    config_path = os.path.join(args.results_dir, 'ttlm_config.txt')
    with open(config_path, 'w') as f:
        f.write(str(config))

    for task_name in task_names:
        task = get_task(task_name)()
        rm = TTTLM(pile_client, model, tokenizer, task, num_fewshot=args.num_fewshot, config=config)
        results_path = os.path.join(args.results_dir, '%s_%d.pth' % (task_name, args.rank))
        eval_tttlm(rm, args.rank, args.world_size, results_path, val_portion=args.val_portion)
