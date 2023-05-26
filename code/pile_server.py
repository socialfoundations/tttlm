"""Server for Pile index."""

import socket
import logging
import argparse

from multiprocessing import Process, Queue
from multiprocessing.connection import Listener

from pile_index import build_roberta_index
from pile_index import split_index_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_servers', type=int, default=1)
    parser.add_argument('--password', type=str, default='ReTraP server.')
    parser.add_argument('--address_path', type=str, default='servers/addresses.txt')
    parser.add_argument('--data_file', type=str, default='pile/train/00.jsonl')
    parser.add_argument('--logging_level', type=str, default='DEBUG')
    parser.add_argument('--timeout', type=int, default=10)
    return parser.parse_args()


class ConnectionHandler(Process):
    """Process to accept and queue incoming connections."""
    
    def __init__(self, listener, queue):
        super().__init__()
        self._listener = listener
        self._queue = queue

    def run(self):
        while(True):
            connection = self._listener.accept()
            address = self._listener.last_accepted
            self._queue.put((connection, address))


class PileServer(Process):
    """Server wrapper around Pile database."""

    def __init__(self, address_path, password, pile_index,
                                               server_name='pile_server',
                                               logging_level=logging.DEBUG,
                                               timeout=20):
        """Initialize server."""
        super().__init__()
        self._address_path = address_path
        self._password = password
        self._pile_index = pile_index
        self._server_name = server_name
        self._logging_level = logging_level
        self._timeout = timeout

    def _step(self):
        """Listen for and respond to a single request."""

        connection, address = self._queue.get()
        logging.debug(f'{self._server_name} accepted connection from '\
                      f'{address}')
        try:
            if connection.poll(self._timeout):
                query = connection.recv()
            else:
                logging.warning(f'{self._server_name} timed out waiting for '\
                                'query. Closing connection.')
                connection.close()
                return True
        except Exception as e:
            logging.warning(f'{self._server_name} failed to receive query: '\
                            f'{e}\n Closing connection.')
            connection.close()
            return True

        if query == '_SHUTDOWN_SERVER_':
            logging.info(f'Shutting down {self._server_name} at: '\
                         f'{self._address}')
            self._listener.close()
            connection.close()
            return False

        result = self._pile_index.vector_query(*query)
        try:
            connection.send(result)
        except Exception as e:
            logging.warning(f'{self._server_name} failed to send result: {e}.')

        connection.close()
        return True

    def run(self):
        """Serve requests on connection."""
        logging.basicConfig(level=self._logging_level)

        # Binding to port 0 will pick an open port
        ipaddr = socket.gethostbyname(socket.gethostname())
        self._listener = Listener((ipaddr, 0), authkey=self._password)
        self._address = self._listener.address
        self._server_name += str(self._address)

        # write ip address and port to file
        with open(self._address_path, 'a') as address_file:
            address_file.write(f'{self._address[0]}:{self._address[1]}\n')

        logging.info(f'{self._server_name} listening for connections at: '\
                     f'{self._address}')

        self._queue = Queue()
        self._conn_handler = ConnectionHandler(self._listener, self._queue)
        self._conn_handler.start()

        while self._step():
            pass

        self._conn_handler.terminate()


if __name__ == '__main__':

    args = parse_args()

    logging.getLogger().setLevel(args.logging_level)

    pile_index = build_roberta_index(args.data_file)
    if args.num_servers > 1:
        pile_indices = split_index_data(pile_index, args.num_servers)
    else:
        pile_indices = [pile_index]

    for i, pile_index in enumerate(pile_indices):
        server_name = 'Server-' + args.data_file.split('.')[0] + f'-{i}'
        server = PileServer(args.address_path, args.password.encode('utf-8'),
                                               pile_index,
                                               server_name,
                                               args.logging_level,
                                               args.timeout)
        server.start()