"""Utility methods for building nearest neighbor index for Pile."""

import os
import json
import logging
import argparse
from tqdm import tqdm

import faiss
import numpy as np

from text_embedding import RobertaEmbedding
from text_embedding import GPTNeoEmbedding


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model_checkpoint', type=str,
                        default='models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000')
    parser.add_argument('--data_file', type=str, default='00.jsonl')
    parser.add_argument('--data_dir', type=str, default='pile/train')
    parser.add_argument('--output_dir', type=str, default='indexes/roberta-large')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=8000000)
    return parser.parse_args()


def add_batch_to_index(text_batch, model, index):
    """Add a batch of texts to the index."""
    batch_embeddings = model(text_batch).cpu().numpy()
    if np.isnan(batch_embeddings).any():
        batch_embeddings = np.nan_to_num(batch_embeddings).astype(np.float32)
        logging.warning("NaN encountered in text: %s", text_batch)
        logging.warning("Replacing NaN with 0.0.")
    index.add(batch_embeddings)


def slice_datafile(data_path: str, begin=0, end=1E7):
    """Take slice from data file.
    
    Parameters
    ----------
    data_path : str
        Path to data file.
    begin : int, optional
        Start index of slice, by default 0
    end : int, optional
        End index of slice, by default 1E7
        
    Returns
    -------
    None
        Writes slice to file 'data_path-begin-end'."""

    logging.info("Reading text file: %s", data_path)
    lines = []
    with open(data_path, 'r') as json_lines:
        for (counter, line) in tqdm(enumerate(json_lines)):
            if counter >= begin and counter < end:
                lines.append(line)
            if counter >= end:
                break
    
    data_slice_file = data_path + '-' + str(begin) + '-' + str(end) 
    logging.info("Writing slice to file: %s", data_slice_file)
    with open(data_slice_file, 'w') as data_slice:
        data_slice.writelines(lines)


def split_datafile(data_path: str, slice_size : int = 500000):
    """Split data file into slices.
    
    Parameters
    ----------
    data_path : str
        Path to data file.
    slice_size : str, optional
        Size of slices, by default 500000
        
    Returns
    -------
    None
        Writes slices to files. """

    logging.info("Reading text file: %s", data_path)
    with open(data_path, 'r') as json_lines:
        lines = json_lines.readlines()
    
    for i in range(0, len(lines), slice_size):
        data_slice_file = data_path + '-' + str(i) + '-' + str(i + slice_size)
        logging.info("Writing slice to file: %s", data_slice_file)
        with open(data_slice_file, 'w') as data_slice:
            data_slice.writelines(lines[i:i + slice_size])


def build_pile_database(data_path: str, index_path: str, embedding_model, batch_size: int = 1, begin=0, end=1E7):
    """Build nearest neighbor database for Pile data.
    
    Build index for a subset of the Pile data given by interval [begin, end).

    Parameters
    ----------
    data_path : str
        Path to data file.
    index_path : str
        Path to index file.
    embedding_model : TextEmbedding
        Embedding model.
    batch_size : int, optional
        Batch size, by default 1
    begin : int, optional
        Start index of slice, by default 0
    end : int, optional
        End index of slice, by default 1E7
    
    Returns
    -------
    None

    See Also
    --------
        [FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
    """

    logging.info("Creating FAISS IndexFlatL2 index.")
    index = faiss.IndexFlatL2(embedding_model.embedding_dimension)

    # Read text items
    logging.info("Reading text file: %s", data_path)
    with open(data_path, 'r') as json_lines:

        text_batch = []
        for (counter, line) in tqdm(enumerate(json_lines)):
            if counter >= begin and counter < end:
                text_item = json.loads(line)['text'] 
                # map empty text to single whitespace
                if not text_item:
                    logging.warning("Empty text encountered at line %d.", counter)
                    text_item = ' '
                text_batch.append(text_item)
                if len(text_batch) == batch_size:
                    add_batch_to_index(text_batch, embedding_model, index)
                    text_batch = []
            if counter >= end:
                break

        # handle last batch
        if len(text_batch) > 0:
            add_batch_to_index(text_batch, embedding_model, index)

    # Write index
    logging.info("Writing FAISS index.")
    faiss.write_index(index, index_path)


if __name__ == '__main__':

    args = parse_args()

    logging.getLogger().setLevel('INFO')

    data_path = os.path.join(args.data_dir, args.data_file)
    logging.info("Data path: %s", data_path)

    os.makedirs(args.output_dir, exist_ok=True)
    index_model_file = os.path.join(args.output_dir, 'embedding_model.txt')
    with open(index_model_file, 'w') as f:
        f.write(args.embedding_model_checkpoint)

    if args.begin == 0 and args.end > 7500000:
        # take whole file
        index_path = data_path + '.index'
    else:
        index_path = data_path + '-' + str(args.begin) + '-' + str(args.end) + '.index'

    logging.info("Index file: %s", index_path)

    if args.embedding_model_checkpoint == 'EleutherAI/gpt-neo-1.3B':
        embedding_model = GPTNeoEmbedding(args.device)
    else:
        embedding_model = RobertaEmbedding(args.embedding_model_checkpoint, args.device)

    logging.info("Building Pile index.")
    build_pile_database(data_path, index_path, embedding_model, args.batch_size, args.begin, args.end)
    logging.info("Pile index built.")
