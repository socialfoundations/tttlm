"""Client for PileServer."""

import os
import time
import logging
import argparse

import random
import json
import faiss

import multiprocess
from multiprocessing.connection import Client

from pile_index import PileIndex
from text_embedding import RobertaEmbedding


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--address_path', type=str, default='servers/addresses.txt')
    parser.add_argument('--password', type=str, default='ReTraP server.')
    parser.add_argument('--embedding_model_checkpoint', type=str,
                        default='models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000')
    parser.add_argument('--query', type=str, default='hi')
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--probe_servers', action='store_true')
    parser.add_argument('--shutdown', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--logging_level', type=str, default='INFO')
    return parser.parse_args()


class PileClient:
    """Send queries to PileServer."""

    def __init__(self, address_path, password, embedding_model=None, timeout=20):
        """Connect to server and send request.

        Parameters
        ----------
        address_path: str
            Path to file containing addresses of servers
        password: bytes
            Password for authentication
        embedding_model: TextEmbedding
            Embedding model to use for string queries
        timeout: int
            Timeout in seconds
        """

        self._address_path = address_path
        self._password = password
        self._timeout = timeout

        self.embedding_model = embedding_model
        if self.embedding_model is not None:
            assert hasattr(self.embedding_model, 'embedding_dimension')

    def _fetch_results(self, query):
        """Fetch results from multiple servers with timeout.

        Parameters
        ----------
        query: (np.ndarray, int)
            Query to send to servers, pair of vector and number of neighbors

        Returns
        -------
        List[Tuple[np.ndarray, List[str]]]
            List of results from servers.
        """

        addresses = get_addresses_from_file(self._address_path)

        connections = []
        for address in addresses:
            try:
                client = Client(address, authkey=self._password)
                connections.append(client)
            except:
                logging.warning('Connection failed at: %s', address)
        logging.info('Connected to %d servers.', len(connections))

        for connection in connections:
            try:
                connection.send(query)
            except:
                logging.warning('Failed to send query to: %s', connection)
                connection.close()
                connections.remove(connection)
        logging.info('Sent query to %d servers.', len(connections))

        results = []
        for connection in connections:
            try:
                if connection.poll(timeout=self._timeout):
                    results.append(connection.recv())
                else:
                    logging.warning('Timeout exceeded.')
            except Exception as e:
                logging.warning('Failed to receive result: %s', e)

            connection.close()
        logging.info('Received results from %d servers.', len(results))

        return results

    def string_query(self, query_string: str, num_neighbors: int):
        """Nearest neighbor string query.
        
        Parameters
        ----------
        query_string: str
            String to query
        num_neighbors: int
            Number of neighbors to return
        
        Returns:
        --------
        np.ndarray, List[str]
            PileIndex vector query results, pair of vectors and data items
        """

        assert self.embedding_model
        query_vector = self.embedding_model([query_string]).cpu().numpy()
        query = (query_vector, num_neighbors)
        results = self._fetch_results(query)
    
        dimension = results[0][0].shape[1]
        results_index = faiss.IndexFlatL2(dimension)
    
        results_items = []
        for vectors, data_items in results:
            results_index.add(vectors)
            results_items += data_items
        results_data_dict = {i: item for (i, item) in enumerate(results_items)}
    
        index = PileIndex(results_index, results_data_dict)
        return index.vector_query(query_vector, num_neighbors)


def get_addresses_from_file(address_path):
    """Get list of IP addresses and ports from file.

    Parameters
    ----------
    address_path: str
        Location of file with IP addresses and ports

    Returns
    -------
    List[Tuple[str, int]]
        List of IP addresses and port"""

    assert os.path.exists(address_path)

    with open(address_path, 'r') as address_path:
        address_lines = address_path.readlines()

    addresses = []
    for line in address_lines:
        ip, port = line.strip().split(':') 
        addresses.append((ip, int(port)))

    return addresses


def probe_servers(address_path, password):
    """Check which servers are alive.

    Parameters
    ----------
    address_path: str
        Location of file with IP addresses and ports
    password: bytes
        Password for authentication

    Returns
    -------
    List[Tuple[str, int]]
        List of IP addresses and port that are alive
    """

    addresses = get_addresses_from_file(address_path)
    times = []
    alive = []
    for address in addresses:
        try:
            start = time.time()
            client = Client(address, authkey=password)
            if client.writable:
                logging.info('Connection successful at: %s', address)
                alive.append(address)
            else:
                logging.warning('Cannot write to: %s', address)
            client.close()
            times.append(time.time() - start)
        except:
            logging.warning('Connection failed at: %s', address)
            continue
    logging.info('Connection successful at %d/%d servers.', len(alive),
                 len(addresses))
    logging.info('Average connection time: %.3f s.', sum(times) / len(times))

    return alive


def shutdown_servers(address_path, password):
    """Try to shutdown all servers found in address_path."""

    if os.path.exists(address_path):
        addresses = get_addresses_from_file(address_path)
        os.remove(address_path)

        for address in addresses:
            try:
                client = Client(address, authkey=password)
                client.send('_SHUTDOWN_SERVER_')
                client.close()
            except:
                logging.warning('Connection failed at: %s', address)
                continue


def roberta_client(address_path='addresses.txt',
                   password=b'ReTraP server.',
                   embedding_model_checkpoint='models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000'):
    """Return PileClient with Roberta embedding model."""
    embedding_model = RobertaEmbedding(embedding_model_checkpoint, 'cuda')
    return PileClient(address_path, password, embedding_model=embedding_model)


def _test_server(address_path, password, num_queries=1000):
    """Test server with random one nearest neighbor queries."""

    client = roberta_client(address_path, password)
    data_path = 'pile/train/01.jsonl'

    logging.debug('Reading data from %s', data_path)
    with open(data_path, 'r') as data_file:
        lines = data_file.readlines()
    random.seed(0)
    random.shuffle(lines)

    correct = 0
    total_query_time = 0.0

    for line in lines[:num_queries]:
        text = json.loads(line)['text']
        start_time = time.time()
        _, data = client.string_query(text, 1)
        retrieved = data[0]
        query_time = time.time() - start_time
        total_query_time += query_time
        logging.info('Query time: %.2f', query_time)
        if text != retrieved:
            logging.warning('Retrieval incorrect.')
            logging.warning('Query: %s', text[:100])
            logging.warning('Retrieved: %s', retrieved[:100])
        else:
            correct += 1
            logging.info('Retrieval correct.')
    
    logging.info('Accuracy: %.2f', correct / num_queries)
    logging.info('Average query time: %.2f', total_query_time / num_queries)


def _test_server_parallel_queries(address_path, password, num_queries=1000):
    """Test server with parallel queries."""

    client = roberta_client(address_path, password)
    data_path = 'pile/train/01.jsonl'

    logging.debug('Reading data from %s', data_path)
    with open(data_path, 'r') as data_file:
        lines = data_file.readlines()
    random.seed(0)
    random.shuffle(lines)

    texts = [json.loads(line)['text'] for line in lines[:num_queries]]

    def query(text):
        logging.info('Querying: %s', text[:50])
        _, data = client.string_query(text, 1)
        return text == data[0]

    p = multiprocess.get_context('spawn').Pool()
    start_time = time.time()
    correct = p.map(query, texts)
    end_time = time.time()

    logging.info('Accuracy: %.2f', sum(correct) / num_queries)
    logging.info('Average query time: %.2f', (end_time - start_time) / num_queries)


if __name__ == '__main__':

    args = parse_args() 

    logging.getLogger().setLevel(args.logging_level)

    password = args.password.encode('utf-8')

    if args.probe_servers:
        probe_servers(args.address_path, password)
        exit()

    if args.shutdown:
        shutdown_servers(args.address_path, password)
        exit()

    if args.test:
        _test_server_parallel_queries(args.address_path, password)
        _test_server(args.address_path, password)
        exit()

    client = roberta_client(args.address_path, password, args.embedding_model_checkpoint)

    start_time = time.time()
    _, texts = client.string_query(args.query, args.num_neighbors)
    logging.info('Query time: %.2f', time.time() - start_time)
    print(texts)
