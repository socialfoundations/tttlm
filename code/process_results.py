"""Process and plot evaluation results."""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

from lm_eval.tasks import get_task
from utils import aggregate

from utils import PILE_WEIGHTS


colors = { 'neutral' : 'steelblue',
           'good' : 'seagreen',
           'bad' : 'firebrick' }


# six largest tasks, more than 70% of the weight
# weights come from PILE_WEIGHTS
pile_top6 = [('pile_github', 7.59),
             ('pile_arxiv', 8.96),
             ('pile_openwebtext2', 10.01),
             ('pile_books3', 12.07),
             ('pile_pubmed-central', 14.4),
             ('pile_pile-cc', 18.11)]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--results_dir2', type=str, default=None)
    parser.add_argument('--results_dir3', type=str, default=None)
    parser.add_argument('--world_size', type=int, default=32)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--error_bars', action='store_true')
    return parser.parse_args()


def plot_curve(results_dir, aggregate_stats, task_name, metric_name, xs, xlabel='neighbors'):
    """Plot curve for individual task."""
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
    """Plot curves for top 6 tasks."""
    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(1, 6, figsize=(8, 1.6))
    task_names = [x[0] for x in pile_top6]
    for task_name, ax in list(zip(task_names, axs.flatten())):
        xs = list(range(len(aggregate_stats[task_name])))
        ys = [aggregate_stats[task_name][x][metric_name] for x in xs]
        if task_name == task_names[0]:
            ax.set_ylabel(metric_name)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.plot(xs, ys, linewidth=2, color=colors['neutral'])
        ax.title.set_fontsize(10)
        ax.title.set_text(task_name[5:])
        ax.set_xticks([0, 25, 50])
        ax.set_xlabel('neighbors')
    plt.tight_layout()
    plt.savefig('%s/%s-top.pdf' % (results_dir, metric_name))
    plt.close()


def pile_all_error_bars(results_dir):
    """Compute weighted error bars for pile_all task."""
    task_names = list(PILE_WEIGHTS.keys())
    befores = []
    afters = []
    for task_name in task_names:
        before_err, after_err = pile_task_error_bars(results_dir, task_name)
        befores.append(before_err)
        afters.append(after_err)
    task_weights = [PILE_WEIGHTS[x]/100 for x in task_names]
    before_err = np.average(befores, axis=0, weights=task_weights)
    after_err = np.average(afters, axis=0, weights=task_weights)
    return before_err, after_err


def pile_task_error_bars(results_dir, task_name):
    """Load error bars for basic pile task."""
    bootstrap = torch.load('%s/bootstrap_%s.pth' % (results_dir, task_name))
    bootstrap_before = bootstrap[0]
    bootstrap_after = bootstrap[-1]
    before_err = np.quantile(bootstrap_before, [0.1, 0.9])
    after_err = np.quantile(bootstrap_after, [0.1, 0.9])
    return before_err, after_err


def load_bootstrap_error_bars(results_dir, task_name):
    if task_name == 'pile_all':
        return pile_all_error_bars(results_dir)
    else:
        return pile_task_error_bars(results_dir, task_name)


def plot_before_after_standalone(results_dir, aggregate_stats, task_name, metric_name, error_bars=False):
    """Plot before-after bar chart for individual task."""
    metrics_before = aggregate_stats[task_name][0]
    metrics_after = aggregate_stats[task_name][-1]
    plt.figure(figsize=((2, 3)))
    plt.title(task_name[5:])
    before = metrics_before[metric_name]
    after = metrics_after[metric_name]
    if after > before:
        after_color = colors['bad']
    else:
        after_color = colors['good']
    if error_bars:
        before_err, after_err = load_bootstrap_error_bars(results_dir, task_name)
        plt.bar(['before', 'after'], [before, after],
                     yerr=[[before - before_err[0], after - after_err[0]],
                           [before_err[1] - before, after_err[1] - after]],
                           color=[colors['neutral'], after_color],
                           capsize=5)
    else:
        plt.bar(['before', 'after'], [before, after], label=task_name,
                                                  color=[colors['neutral'], after_color])
        plt.text(1, after, '%.0f %%' % (100 * after/before), ha='center', va='bottom')
    plt.yticks(fontsize=9)
    plt.tight_layout()
    if error_bars:
        plt.savefig('%s/before-after-%s-error-bars.pdf' % (results_dir, task_name))
    else:
        plt.savefig('%s/before-after-%s.pdf' % (results_dir, task_name))
    plt.close()


def plot_before_after_top(results_dir, aggregate_stats, metric_name, error_bars=False):
    """Plot before-after bar chart for top 6 tasks."""
    # plot 8 x 4 grid of figures
    _, axs = plt.subplots(1, 6, figsize=(8, 1.6))
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
        before = metrics_before[metric_name]
        after = metrics_after[metric_name]
        if error_bars:
            before_err, after_err = load_bootstrap_error_bars(results_dir, task_name)
            ax.bar(['before', 'after'], [before, after],
                    yerr=[[before - before_err[0], after - after_err[0]],
                          [before_err[1] - before, after_err[1] - after]],
                          color=[colors['neutral'], color],
                          capsize=5)
        else:
            ax.bar(['before', 'after'], [before, after],
                color=[colors['neutral'], color])
            ax.text(1, metrics_after[metric_name], '%.0f %%' % (100 * metrics_after[metric_name]/metrics_before[metric_name]),
                ha='center', va='bottom')
        ax.set_ylim([0, np.max([before*1.15, after*1.15])])
        # change xtick font size
    plt.tight_layout()
    if error_bars:
        plt.savefig('%s/before-after-top-error_bars.pdf' % (results_dir))
    else:
        plt.savefig('%s/before-after-top.pdf' % (results_dir))
    plt.close()


def plot_before_after_all(results_dir, aggregate_stats, task_names, metric_name, error_bars):
    """Plot before-after bar chart for all tasks."""
    # plot 8 x 4 grid of figures
    _, axs = plt.subplots(4, 6, figsize=(8, 6))
    plt.rcParams.update({'font.size': 8})
    for task_name, ax in list(zip(task_names, axs.flatten())):
        metrics_before = aggregate_stats[task_name][0]
        metrics_after = aggregate_stats[task_name][-1]
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_title(task_name[5:])
        if metrics_after[metric_name] > metrics_before[metric_name]:
            color = colors['bad']
        else:
            color = colors['good']
        before = metrics_before[metric_name]
        after = metrics_after[metric_name]
        print (task_name, before, after)
        if error_bars:
            before_err, after_err = load_bootstrap_error_bars(results_dir, task_name)
            ax.bar(['before', 'after'], [before, after],
                    yerr=[[before - before_err[0], after - after_err[0]],
                          [before_err[1] - before, after_err[1] - after]],
                          color=[colors['neutral'], color],
                          capsize=5)
        else:
            ax.bar(['before', 'after'], [before, after], color=[colors['neutral'], color])
            ax.text(1, metrics_after[metric_name], '%.0f %%' % (100 * metrics_after[metric_name]/metrics_before[metric_name]),
                ha='center', va='bottom')
        ax.set_ylim([0, np.max([before*1.15, after*1.15])])
    axs[-1, -1].axis('off')
    plt.tight_layout()
    if error_bars:
        plt.savefig('%s/before-after-all-error-bars.pdf' % (results_dir))
    else:
        plt.savefig('%s/before-after-all.pdf' % (results_dir))
    plt.close()


def plot_training_costs(results_dir, aggregate_stats, task_names):
    """Plot training costs for all tasks."""
    plt.rcParams.update({'font.size': 12})
    task_costs = []
    for task_name in task_names:
        costs = [x['training_cost'] for x in aggregate_stats[task_name]]
        costs = [x for x in costs if not np.isnan(x)]
        task_costs.append(costs)
    plt.figure(figsize=((12, 4)))
    plt.yscale('log')
    plt.ylabel('Training cost (sec)')
    plt.boxplot(task_costs, labels=task_names)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('%s/training-costs.pdf' % (results_dir))
    plt.close()


def plot_comparisons(results_dir, aggregate_stats1, aggregate_stats2, aggregate_stats3, task_name):
    """Plot before-after bar chart for all models."""

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
    plt.close()


def compute_pile_all(aggregate_stats, metric='bits_per_byte'):
    """Compute weighted aggregate for synthetic pile_all task."""
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


def bootstrap(data, task_name, n_resamples=1000):
    """Compute bootstrap of aggregate statistics"""
    n = len(data)
    task = get_task(task_name)(download=False)
    statistics = []
    for _ in tqdm(range(n_resamples)):
        sample = data[np.random.randint(0, n, n)]
        statistics.append(aggregate(sample, task)['bits_per_byte'])
    return np.array(statistics)


def compute_bootstrap_error_bars(results_dir, metrics, task_name):
    """Compute error bars using bootstrap."""
    before = metrics[:, 0]
    after = metrics[:, -1]
    bootstrap_before = bootstrap(before, task_name)
    bootstrap_after = bootstrap(after, task_name)
    bootstrap_values = [bootstrap_before, bootstrap_after]
    torch.save(bootstrap_values, '%s/bootstrap_%s.pth' % (results_dir, task_name))


def compute_aggregate_stats(task_names, results_dir, world_size, bootstrap=False):
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
        if len(metrics) == 0:
            print('No results for %s' % (task_name))
            continue
        if len(losses) == 0:
            print('No losses for %s' % (task_name))
            continue
        if len(training_costs) == 0:
            print('No training costs for %s' % (task_name))
            continue
        median = int(np.median([len(m) for m in metrics]))
        metrics = [metrics[i] for i in range(len(metrics)) if len(metrics[i]) == median]
        median = int(np.median([len(l) for l in losses]))
        losses = [losses[i] for i in range(len(losses)) if len(losses[i]) == median]
        median = int(np.median([len(l) for l in training_costs]))
        training_costs = [training_costs[i] for i in range(len(training_costs)) 
                          if len(training_costs[i]) == median]

        # 2d array (num_points, num_steps) where each entry is a dict of metrics
        metrics = np.array(metrics)
        if bootstrap:
            compute_bootstrap_error_bars(results_dir, metrics, task_name)

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

    args = parse_args()

    task_names = list(PILE_WEIGHTS.keys())
    aggregate_stats = compute_aggregate_stats(task_names, args.results_dir,
                                              args.world_size, bootstrap=args.bootstrap)
    if args.results_dir2 is not None:
        aggregate_stats2 = compute_aggregate_stats(task_names,
                                                   args.results_dir2, args.world_size)
    if args.results_dir3 is not None:
        aggregate_stats3 = compute_aggregate_stats(task_names,
                                                   args.results_dir3, args.world_size)
    task_names = ['pile_all'] + task_names

    if args.results_dir2 is not None and args.results_dir3 is not None:
        for task_name in task_names:
            plot_comparisons(args.results_dir, aggregate_stats,
                             aggregate_stats2, aggregate_stats3, task_name)

    plot_curve_top(args.results_dir, aggregate_stats, 'bits_per_byte')
    plot_curve_top(args.results_dir, aggregate_stats, 'byte_perplexity')
    plot_curve_top(args.results_dir, aggregate_stats, 'word_perplexity')
    plot_before_after_top(args.results_dir, aggregate_stats, 'bits_per_byte', args.error_bars)
    plot_before_after_all(args.results_dir, aggregate_stats, task_names, 'bits_per_byte',
                          args.error_bars)
    plot_training_costs(args.results_dir, aggregate_stats, task_names)

    for task_name in task_names:

        plot_before_after_standalone(args.results_dir, aggregate_stats, task_name,
                                     'bits_per_byte', args.error_bars)

        xs = range(len(aggregate_stats[task_name]))
        plot_curve(args.results_dir, aggregate_stats, task_name, 'bits_per_byte', xs,
                   xlabel='neighbors')

        xs = xs[1:]
        plot_curve(args.results_dir, aggregate_stats, task_name, 'training_loss', xs,
                   xlabel='neighbors')

        print(task_name, len(xs))
