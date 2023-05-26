"""Interpolation baseline."""

import os
import logging
import argparse
import random

from tqdm import tqdm
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForCausalLM

from pile_client import roberta_client

from lm_eval_interpolation.gpt2 import HFLM
from lm_eval.tasks import get_task, ALL_TASKS


@dataclass
class TTTLMConfig:
    """TTTLM configuration."""
    model : str ='EleutherAI/gpt-neo-1.3B'
    tokenizer : str ='EleutherAI/gpt-neo-1.3B'
    num_neighbors : int = 20
    num_iters : int = 20
    reset_weights : bool = True
    max_length : int = 2048
    stride : int = 2048
    mask_probability : float = 0.15
    learning_rate : float = 7e-6
    adam_epsilon : float = 1e-8
    distance_threshold : float = 4.0


class TTTLM:
    """Test time training on nearest neighbors."""

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

    def _evaluate(self, eval_doc, texts_distances):
        """Evaluate using calls to lm_eval."""
        self.eval_model.gpt2.eval()

        if isinstance(eval_doc, str):
            doc, ctx = eval_doc, ''
        else:
            doc, ctx = eval_doc

        reqs = self.task.construct_requests(doc, ctx)
        if not isinstance(reqs, (list, tuple)):
            reqs = [reqs]

        assert(len(reqs) == 1)
        
        for req in reqs:
            reqtype = req.request_type
            resp = getattr(self.eval_model, reqtype)([req.args], texts_distances=texts_distances)
            
        all_metrics = []
        for one_resp in resp:
            resps = [one_resp]
            resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]
            metrics = self.task.process_results(doc, resps)
            all_metrics.append(metrics)

        return all_metrics

    def _filter_retrieved(self, query_text, vectors, texts):
        """Filter retrieved texts based on distance to query embedding."""
        query_vector = self.pile_client.embedding_model([query_text]).cpu().numpy()
        distances = np.linalg.norm(vectors - query_vector, axis=1)
        # near = distances < self.config.distance_threshold
        # select texts where near is true
        # texts = [text for text, n in zip(texts, near) if n]
        return texts, distances

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
        # On long runs, retrieval sometimes fails due to network issues.
        try:
            vectors, texts = self.pile_client.string_query(query_text,
                                                    self.config.num_neighbors)
        except Exception as e:
            logging.warning("Failed to retrieve: %s", e)
            texts = []

        return self._filter_retrieved(query_text, vectors, texts)


def eval_tttlm(tttlm, rank, world_size, results_path, val_portion=1.0):
    """Evaluate TTTLM on test set. """

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

    all_results = []
    for i, doc in tqdm(enumerate(task_docs)):
        texts_distances = tttlm._retrieve(doc)
        all_results.append(tttlm._evaluate(doc, texts_distances))

    results = (all_results, None, None, None)
    torch.save(results, results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--tokenizer', type=str, default='EleutherAI/gpt-neo-1.3B')
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B')
    parser.add_argument('--tasks', type=str, default='pile_all')
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--val_portion', type=float, default=1.0)
    parser.add_argument('--num_neighbors', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=2048)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--address_path', type=str, default='servers/addresses.txt')

    args = parser.parse_args()
    config = TTTLMConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        num_neighbors=args.num_neighbors,
        max_length=args.max_length,
        stride=args.stride,
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

    pile_client = roberta_client(address_path=args.address_path)

    os.makedirs(args.results_dir, exist_ok=True)
    config_path = os.path.join(args.results_dir, 'tttlm_config.txt')
    with open(config_path, 'w') as f:
        f.write(str(config))

    for task_name in task_names:
        task = get_task(task_name)()
        rm = TTTLM(pile_client, model, tokenizer, task, 
                         num_fewshot=args.num_fewshot, config=config)
        results_path = os.path.join(args.results_dir, '%s_%d.pth' % (task_name, args.rank))
        eval_tttlm(rm, args.rank, args.world_size, results_path, val_portion=args.val_portion)
