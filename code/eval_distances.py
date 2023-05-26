"""Evaluating nearest neighbor distances."""

import json
import logging

import numpy as np
from tqdm import tqdm

import os
import multiprocessing
import argparse

from text_embedding import RobertaEmbedding

from pile_client import PileClient
from pile_client import get_addresses_from_file


def distances_to_nns(address_path, results_dir, num_queries=10000, num_neighbors=100, num_addresses=180):
    """Compute distances to nearest neighbors."""

    multiprocessing.log_to_stderr(logging.DEBUG)

    data_path = 'pile/val.jsonl'
    logging.debug('Reading data from %s', data_path)

    lines = []
    with open(data_path, 'r') as data_file:
        for _ in range(num_queries):
            lines.append(data_file.readline())

    addresses = get_addresses_from_file(address_path)
    addresses = addresses[:num_addresses]

    model_checkpoint = 'models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000'
    embedding_model = RobertaEmbedding(model_checkpoint, 'cuda')

    client = PileClient(addresses, b'ReTraP server.', embedding_model=embedding_model)

    distances_list = []
    text_embeddings = []
    for line in tqdm(lines):
        text = json.loads(line)['text']
        text_embedding = embedding_model(text).cpu().numpy()
        text_embeddings.append(text_embedding)
        vectors, _ = client.string_query(text, num_neighbors)
        distances = np.sort(np.sqrt(((vectors - text_embedding)**2).sum(axis=1)))
        distances_list.append(distances)

    random_distances_list = []
    for _ in range(num_queries):
        [i, j] = np.random.randint(0, num_queries, 2)
        distance = np.linalg.norm(text_embeddings[i] - text_embeddings[j])
        random_distances_list.append(distance)

    file_name = 'distances_to_nns_{}_{}_{}.txt'.format(num_queries, num_neighbors, num_addresses)
    np.savetxt(os.path.join(results_dir, file_name) , np.array(distances_list))

    file_name = 'distances_random_{}.txt'.format(num_queries)
    np.savetxt(os.path.join(results_dir, file_name), np.array(random_distances_list))

    return distances_list, random_distances_list


if __name__ == '__main__':

    logging.getLogger().setLevel('INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_addresses', type=int, default=180)
    parser.add_argument('--address_path', type=str, default='servers/addresses.txt')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    logging.info('Number of addresses: %d', args.num_addresses)
    distances_to_nns(args.address_path, args.results_dir, num_addresses=args.num_addresses)
