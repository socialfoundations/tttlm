"""Merge results from distributed evaluation."""

import torch
import argparse

from lm_eval.tasks import get_task, ALL_TASKS
from utils import aggregate


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--tasks', type=str, default='pile_all')
    parser.add_argument('--world_size', type=int, default=32)
    args = parser.parse_args()

    if args.tasks == 'pile_all':
        from utils import PILE_WEIGHTS
        task_names = PILE_WEIGHTS.keys()
    elif args.tasks == 'super_glue':
        task_names = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    else:
        from utils import pattern_match
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    metrics = {}
    for task_name in task_names:
        print(task_name)
        all_stats = []
        task = get_task(task_name)(download=False)
        for rank in range(args.world_size):
            results_file = '%s/%s_%d.pth' % (args.results_dir, task_name, rank)
            try:
                results = torch.load(results_file)
                all_stats += results[0]
            except:
                print('Not found: %s' % (results_file))
        metrics[task_name] = aggregate(all_stats, task)
    
    for task_name in task_names:
        print('%s %.2f' % (task_name, metrics[task_name]['bits_per_byte']))
        