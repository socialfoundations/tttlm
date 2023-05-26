"""Merge results from distributed evaluation."""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from lm_eval.tasks import get_task, ALL_TASKS
from utils import aggregate

from utils import PILE_WEIGHTS


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

    metrics_before = {}
    metrics_after = {}

    pile_all = 0

    for task_name in task_names:
        all_stats = []
        all_losses = []
        training_costs = []
        retrieval_costs = []

        task = get_task(task_name)(download=False)
        for rank in tqdm(range(args.world_size)):
            results_file = '%s/%s_%d.pth' % (args.results_dir, task_name, rank)
            try:
                results = torch.load(results_file)
                all_stats += results[0]
            except:
                print('Not found: %s' % (results_file))

        all_stats = np.array(all_stats)
        aggregate_stats = []
        for j in range(all_stats.shape[1]):
            aggregate_row = all_stats[:, j]
            aggregate_stats.append(aggregate(aggregate_row, task))

        plot_stats = []
        metric_name = list(aggregate_stats[0].keys())[2]
        for aggregate_entry in aggregate_stats:
            plot_stats.append(aggregate_entry[metric_name])

        metrics_before[task_name] = plot_stats[0]
        metrics_after[task_name] = min(plot_stats)

        plt.figure()
        plt.plot(np.arange(0, 0.2, 0.01), plot_stats)
        plt.ylabel(metric_name)
        plt.xlabel('alpha')
        plt.savefig('%s/alpha_%s_%s.pdf' % (args.results_dir, task_name, metric_name))

        pile_all += np.array(plot_stats) * PILE_WEIGHTS[task_name]

    for task_name in task_names:
        print(task_name, metrics_before[task_name], metrics_after[task_name])

    for task_name in task_names:
        print('%s %.2f' % (task_name, metrics_after[task_name]))

    print(pile_all)
