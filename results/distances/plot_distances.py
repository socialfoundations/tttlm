"""Plot distances to nearest neighbors."""

import numpy as np
import matplotlib.pyplot as plt


def plot_means(means, labels, legend_title, filename='mean_distances_size.pdf'):
    """Plot means."""
    
    plt.figure(figsize=(7, 4))
    plt.title('Mean distance to nearest neighbor')

    for mean, label in zip(means, labels):
        try:
            alpha = float(label[:-1])/100
        except:
            alpha = 1
        plt.plot(range(1, 101), mean, color='orangered', linewidth=0.2)
        plt.plot(range(1, 101), mean, label=label, color='orangered', alpha=alpha, linewidth=2)
    
    plt.ylabel('Euclidean distance')
    plt.xlabel('Rank of nearest neighbor')
    # put legend to the right of plot
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.legend(title=legend_title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_stddev(mean, stddev, filename='mean_std_distances.pdf'):

    plt.figure(figsize=(5, 4))
    plt.title('Mean distance to nearest neighbor')
    plt.plot(range(1, 101), mean, color='orangered')
    plt.fill_between(range(1, 101), mean+stddev, mean-stddev, color='orangered', alpha=0.25)
    plt.ylabel('Euclidean distance')
    plt.xlabel('Rank of nearest neighbor')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_distribution(distances, filename='distances_hist.pdf', **kwargs):

    plt.figure(figsize=(4, 3))
    plt.title(kwargs.get('title', 'Distances to nearest neighbor'))
    # plot fractional histogram
    plt.hist(distances, bins=100, color='orangered', alpha=0.8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.family': 'serif'})

    files = ['distances_to_nns_10000_100_{}.txt'.format(i) for i in range(18, 181, 18)]
    files = ['distances_to_nns_10000_100_9.txt'] + files
    data = [np.loadtxt(f) for f in files]
    lambada = np.loadtxt('lambada_distances.txt')
    means = [np.mean(d, axis=0) for d in data]
    labels = ['5%'] + [str((i+1)*10)+'%' for i in range(1, 10)]

    lambada_mean = np.mean(lambada, axis=0)
    stddevs = [np.std(d, axis=0) for d in data]
    lambada_std = np.std(lambada, axis=0)

    random_dists = np.loadtxt('distances_random_10000.txt')
    mean_random = np.mean(random_dists)
    max_random = np.max(random_dists)

    print('Total number of distances: {}'.format(len(data[-1][:,0])))

    nn_ds = data[-1][:, 0]
    threshold = 0.0001
    very_close = nn_ds[nn_ds < threshold]
    print('Found {} distances < {}'.format(len(very_close), threshold))

    plot_means(means, labels, 'Index size')
    plot_means([means[-1], lambada_mean], ['pile', 'lambada'], 'Dataset', 'lambada_distances_mean.pdf')
    plot_stddev(means[-1], stddevs[-1])
    plot_stddev(lambada_mean, lambada_std, 'lambada_distances_std.pdf')
    plot_distribution(data[-1][:,0], 'hist_nn.pdf', title='Distances to nearest neighbor')
    plot_distribution(random_dists, 'hist_random.pdf', title='Distances between random queries')
    plot_distribution(very_close, 'hist_close.pdf', title='Distances < 0.0001')
    plot_distribution(lambada[:,0], 'hist_lambada.pdf')

    print(mean_random, max_random)
