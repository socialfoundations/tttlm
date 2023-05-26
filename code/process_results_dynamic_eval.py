"""Process results from dynamic evaluation."""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

from lm_eval.tasks import get_task, ALL_TASKS
from utils import aggregate

from utils import PILE_WEIGHTS


colors = { 'neutral' : 'steelblue',
           'good' : 'seagreen',
           'bad' : 'firebrick' }


# six largest tasks, more than 70% of the weight
pile_top6 = [('pile_github', 7.59),
             ('pile_arxiv', 8.96),
             ('pile_openwebtext2', 10.01),
             ('pile_books3', 12.07),
             ('pile_pubmed-central', 14.4),
             ('pile_pile-cc', 18.11)]


def plot_curve(results_dir, aggregate_stats, task_name, metric_name, xs, xlabel='neighbors', **kwargs):
    """Plot curve."""
    task_stats = aggregate_stats[task_name]
    ys = [task_stats[x][metric_name] for x in xs]
    plt.figure(figsize=((4, 4)))
    plt.title(task_name)
    plt.plot(xs, ys, linewidth=2, color=colors['neutral'])
    plt.xticks(range(0, len(xs), 10))
    plt.ylabel(metric_name)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig('%s/%s_%s.pdf' % (results_dir, task_name, metric_name))
    plt.close()


def plot_curve_top(results_dir, aggregate_stats, metric_name):
    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(1, 6, figsize=(8, 2))
    task_names = [x[0] for x in pile_top6]
    for task_name, ax in list(zip(task_names, axs.flatten())):
        xs = list(range(len(aggregate_stats[task_name])))
        ys = [aggregate_stats[task_name][x][metric_name] for x in xs]
        if task_name == task_names[0]:
            ax.set_ylabel(metric_name)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.plot(xs, ys, linewidth=2, color=colors['neutral'])
        # change title font size
        ax.title.set_fontsize(10)
        ax.title.set_text(task_name[5:])
        # set xticks
        ax.set_xticks([0, 25, 50])
        # set xlabel
        ax.set_xlabel('neighbors')

    plt.tight_layout()
    plt.savefig('%s/%s-top.pdf' % (results_dir, metric_name))
    plt.close()


def plot_before_after(metrics_before, metrics_after, task_name, metric_name):
    """Plot before and after bar plot."""
    plt.title(task_name[5:])
    before = metrics_before[metric_name]
    after = metrics_after[metric_name]
    if after > before:
        after_color = colors['bad']
    else:
        after_color = colors['good']
    plt.bar(['before', 'after'], [before, after], label=task_name,
                                                  color=[colors['neutral'], after_color])
    plt.text(1, after, '%.0f %%' % (100 * after/before), ha='center', va='bottom')
    #plt.ylabel(metric_name)
    # change y tick font size
    plt.yticks(fontsize=9)
    plt.tight_layout()


def plot_before_after_standalone(results_dir, aggregate_stats, task_name, metric_name):
    metrics_before = aggregate_stats[task_name][0]
    metrics_after = aggregate_stats[task_name][6]
    print('%s %.2f' % (task_name, metrics_after['bits_per_byte']))
    plt.figure(figsize=((2, 3)))
    plot_before_after(metrics_before, metrics_after, task_name, metric_name)
    plt.savefig('%s/before-after-%s.pdf' % (results_dir, task_name))
    plt.close()


def plot_before_after_top(results_dir, aggregate_stats, metric_name):
    # plot 8 x 4 grid of figures
    _, axs = plt.subplots(1, 6, figsize=(8, 2))
    task_names = [x[0] for x in pile_top6]
    plt.rcParams.update({'font.size': 8})
    for task_name, ax in list(zip(task_names, axs.flatten())):
        metrics_before = aggregate_stats[task_name][0]
        metrics_after = aggregate_stats[task_name][-1]
        # change font size
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_title(task_name[5:])
        if metrics_after[metric_name] > metrics_before[metric_name]:
            color = colors['bad']
        else:
            color = colors['good']
        ax.bar(['before', 'after'], [metrics_before[metric_name], metrics_after[metric_name]],
                color=[colors['neutral'], color])
        # change xtick font size
        ax.text(1, metrics_after[metric_name], '%.0f %%' % (100 * metrics_after[metric_name]/metrics_before[metric_name]),
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('%s/before-after-top.pdf' % (results_dir))
    plt.close()


def plot_before_after_all(results_dir, aggregate_stats, task_names, metric_name):

    # plot 8 x 4 grid of figures
    _, axs = plt.subplots(4, 6, figsize=(8, 8))
    # change font size
    plt.rcParams.update({'font.size': 8})
    for task_name, ax in list(zip(task_names, axs.flatten())):
        metrics_before = aggregate_stats[task_name][0]
        metrics_after = aggregate_stats[task_name][-1]
        # change font size
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_title(task_name[5:])
        if metrics_after[metric_name] > metrics_before[metric_name]:
            color = colors['bad']
        else:
            color = colors['good']
        ax.bar(['before', 'after'], [metrics_before[metric_name], metrics_after[metric_name]],
                color=[colors['neutral'], color])
        # change xtick font size
        ax.text(1, metrics_after[metric_name], '%.0f %%' % (100 * metrics_after[metric_name]/metrics_before[metric_name]),
                ha='center', va='bottom')
    # avoid empty last panel
    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.savefig('%s/before-after-all.pdf' % (results_dir))
    plt.close()


def plot_training_costs(results_dir, aggregate_stats, task_names):

    plt.rcParams.update({'font.size': 12})

    task_costs = []
    for task_name in task_names:
        costs = [x['training_cost'] for x in aggregate_stats[task_name]]
        costs = [x for x in costs if not np.isnan(x)]
        task_costs.append(costs)

    plt.figure(figsize=((12, 4)))
    # log scale
    plt.yscale('log')
    plt.ylabel('Training cost (sec)')
    plt.boxplot(task_costs, labels=task_names)
    # rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('%s/training-costs.pdf' % (results_dir))
    plt.close()


def plot_comparisons(results_dir, aggregate_stats1, aggregate_stats2, aggregate_stats3, task_name):
    # Compare before and after bits_per_byte for pile_all

    metrics_before1 = aggregate_stats1[task_name][0]
    metrics_after1 = aggregate_stats1[task_name][-1]
    metrics_before2 = aggregate_stats2[task_name][0]
    metrics_after2 = aggregate_stats2[task_name][-1]
    metrics_before3 = aggregate_stats3[task_name][0]
    metrics_after3 = aggregate_stats3[task_name][-1]

    models = ['gpt2', 'gpt2-ttt', 'gpt2-large', 'gpt2-large-ttt', 'gptneo', 'gptneo-ttt']
    metrics = [metrics_before1['bits_per_byte'], metrics_after1['bits_per_byte'],
                metrics_before2['bits_per_byte'], metrics_after2['bits_per_byte'],
                metrics_before3['bits_per_byte'], metrics_after3['bits_per_byte']]

    plt.figure(figsize=(3, 3))
    plt.bar(models, metrics, color=[colors['neutral'], colors['good'], colors['neutral'], colors['good'], colors['neutral'], colors['good']])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Bits per byte')
    plt.title(task_name[5:])
    plt.tight_layout()
    plt.savefig('%s/comparisons_%s.pdf' % (results_dir, task_name[5:]))


def compute_pile_all(aggregate_stats, metric='bits_per_byte'):
    pile_all = []
    task_values = []
    for task in aggregate_stats:
        task_stats = aggregate_stats[task]
        values = [metrics[metric] for metrics in task_stats]
        values = [v * PILE_WEIGHTS[task] / 100 for v in values]
        task_values.append(values)
    median = int(np.median([len(v) for v in task_values]))
    task_values = [v for v in task_values if len(v) == median]
    task_values = np.array(task_values)
    return np.sum(task_values, axis=0)


def compute_aggregate_stats(task_names, results_dir, world_size):
    """Load results from file and aggregate them."""

    if os.path.exists('%s/aggregate_stats.pth' % (results_dir)):
        return torch.load('%s/aggregate_stats.pth' % (results_dir))

    aggregate_stats = {}
    for task_name in task_names:
        metrics = []
        losses = []
        training_costs = []
        retrieval_costs = []

        task = get_task(task_name)(download=False)
        for rank in tqdm(range(world_size)):
            results_file = '%s/%s_%d.pth' % (results_dir, task_name, rank)
            if os.path.exists(results_file):
                print('Found: %s' % (results_file))
                try:
                    results = torch.load(results_file)
                    assert len(results) == 4
                    # length of results[0] is the number of eval points
                    metrics += results[0]
                    losses += results[1]
                    training_costs += results[2]
                    retrieval_costs += results[3]
                except Exception as e:
                    print('Error loading %s' % (results_file))
                    print(e)
                    continue
            else:
                print('Not found: %s' % (results_file))
        
        # Length of all_stats is the number of points evaluated
        # Each element is a list length the number of training steps
        # Filter out the runs that didn't finish
        median = int(np.median([len(m) for m in metrics]))
        metrics = [metrics[i] for i in range(len(metrics)) if len(metrics[i]) == median]
        median = int(np.median([len(l) for l in losses]))
        losses = [losses[i] for i in range(len(losses)) if len(losses[i]) == median]
        median = int(np.median([len(l) for l in training_costs]))
        training_costs = [training_costs[i] for i in range(len(training_costs)) if len(training_costs[i]) == median]

        # 2d array (num_points, num_steps) where each entry is a dict of metrics
        metrics = np.array(metrics)
        losses = np.array(losses)
        training_costs = np.array(training_costs)
        print(metrics.shape)
        print(losses.shape)
        print(training_costs.shape)
        task_stats = []
        try:
            for j in range(metrics.shape[1]):
                task_stats.append(aggregate(metrics[:, j], task))
            # No loss record before training
            task_stats[0]['training_loss'] = np.nan
            task_stats[0]['training_cost'] = np.nan
            for j in range(losses.shape[1]):
                task_stats[j+1]['training_loss'] = np.nanmean(losses[:, j])
            for j in range(training_costs.shape[1]):
                task_stats[j+1]['training_cost'] = np.nanmean(training_costs[:, j])
        except:
            print('Failed on %s' % (task_name))

        aggregate_stats[task_name] = task_stats
    
    pile_all_bpb = compute_pile_all(aggregate_stats, metric='bits_per_byte')
    pile_all_tl = compute_pile_all(aggregate_stats, metric='training_loss')
    pile_all_tc = compute_pile_all(aggregate_stats, metric='training_cost')
    pile_all = [ {'bits_per_byte': bpb, 'training_loss': tl, 'training_cost' : tc} 
                for bpb, tl, tc in zip(pile_all_bpb, pile_all_tl, pile_all_tc) ]
    aggregate_stats['pile_all'] = pile_all

    # pickle the results
    torch.save(aggregate_stats, '%s/aggregate_stats.pth' % (results_dir))

    return aggregate_stats


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({'font.family': 'serif'})

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--results_dir2', type=str, default=None)
    parser.add_argument('--results_dir3', type=str, default=None)
    parser.add_argument('--world_size', type=int, default=32)
    args = parser.parse_args()

    task_names = list(PILE_WEIGHTS.keys())
    aggregate_stats = compute_aggregate_stats(task_names, args.results_dir, args.world_size)
    if args.results_dir2 is not None:
        aggregate_stats2 = compute_aggregate_stats(task_names, args.results_dir2, args.world_size)
    if args.results_dir3 is not None:
        aggregate_stats3 = compute_aggregate_stats(task_names, args.results_dir3, args.world_size)
    task_names = ['pile_all'] + task_names

    if args.results_dir2 is not None and args.results_dir3 is not None:
        plot_comparisons(args.results_dir, aggregate_stats, aggregate_stats2, aggregate_stats3, 'pile_all')
        plot_comparisons(args.results_dir, aggregate_stats, aggregate_stats2, aggregate_stats3, 'pile_github')
        plot_comparisons(args.results_dir, aggregate_stats, aggregate_stats2, aggregate_stats3, 'pile_pubmed-central')

    plot_curve_top(args.results_dir, aggregate_stats, 'bits_per_byte')
    plot_curve_top(args.results_dir, aggregate_stats, 'byte_perplexity')
    plot_curve_top(args.results_dir, aggregate_stats, 'word_perplexity')
    plot_before_after_top(args.results_dir, aggregate_stats, 'bits_per_byte')
    plot_before_after_all(args.results_dir, aggregate_stats, task_names, 'bits_per_byte')
    plot_training_costs(args.results_dir, aggregate_stats, task_names)

    for task_name in task_names:

        plot_before_after_standalone(args.results_dir, aggregate_stats, task_name, 'bits_per_byte')

        xs = range(len(aggregate_stats[task_name]))
        plot_curve(args.results_dir, aggregate_stats, task_name, 'bits_per_byte', xs, xlabel='neighbors')

        xs = xs[1:]
        plot_curve(args.results_dir, aggregate_stats, task_name, 'training_loss', xs, xlabel='neighbors')

        print(task_name, len(xs))
