import fnmatch

from lm_eval.tasks import *

PILE_WEIGHTS = {
	"pile_arxiv": 8.96,
	"pile_bookcorpus2": 0.75,
	"pile_books3": 12.07,
	"pile_dm-mathematics": 1.24,
	"pile_enron": 0.14,
	"pile_europarl": 0.73,
	"pile_freelaw": 6.12,
	"pile_github": 7.59,
	"pile_gutenberg": 2.17,
	"pile_hackernews": 0.62,
	"pile_nih-exporter": 0.30,
	"pile_opensubtitles": 1.55,
	"pile_openwebtext2": 10.01,
	"pile_philpapers": 0.38,
	"pile_pile-cc": 18.11,
	"pile_pubmed-abstracts": 3.07,
	"pile_pubmed-central": 14.40,
	"pile_stackexchange": 5.13,
	"pile_ubuntu-irc": 0.88,
	"pile_uspto": 3.65,
	"pile_wikipedia": 1.53,
	"pile_youtubesubtitles": 0.60
	}

# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def aggregate(results, task):
    """Aggregate results after training."""
    aggregate_results = {}
    for metric in results[0].keys():
        items = []
        for item in results:
            items.append(item[metric])
        aggregate_results[metric] = task.aggregation()[metric](items)
    return aggregate_results

def pile_final(results):
	return {}