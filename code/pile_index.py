"""Nearest neighbor index for text data."""

import os
import json

import faiss
import numpy as np

from tqdm import tqdm


class PileIndex:
    """Nearest neighbor index."""

    def __init__(self, index : faiss.IndexFlatL2, data_dict : dict,
                                                  embedding_model=None):
        """Initialize pile index.
        
        Parameters
        ----------
        index : faiss.IndexFlatL2
            Nearest neighbor index.
        data_dict : dict
            Dictionary mapping index position to data item. 
            data_dict[i] should correspond to the vector at index position i.
        embedding_model : TextEmbedding
            Embedding model for text data.
            Should be identical to the embedding model used to create the index.
        
        Returns
        -------
        PileIndex
            Pile nearest neighbor index.
        """

        self.index = index
        self.data_dict = data_dict
        assert len(self.data_dict) == self.index.ntotal

        self.embedding_model = embedding_model
        if self.embedding_model is not None:
            assert hasattr(self.embedding_model, 'embedding_dimension')

    def vector_query(self, query_vector : np.ndarray, num_neighbors : int):
        """Nearest neighbor vector query.
        
        Parameters
        ----------
        query_vector : np.ndarray
            Vector to query.
        num_neighbors : int
            Number of neighbors to return.
        
        Returns
        -------
        np.ndarray, List[str]
            Pair of vectors and data items.
        """

        assert self.index.d == query_vector.shape[1]

        results = self.index.search_and_reconstruct(query_vector, num_neighbors)
        neighbors = results[1].reshape(num_neighbors)
        vectors = results[2].reshape(num_neighbors, -1)
        data_items = [self.data_dict[i] for i in neighbors]

        return vectors, data_items

    def string_query(self, query_str : str, num_neighbors : int):
        """Nearest neighbor string query.
        
        Parameters
        ----------
        query_str : str
            String to query.
        num_neighbors : int
            Number of neighbors to return.
        
        Returns
        -------
        np.ndarray, List[str]
            Pair of vectors and data items.
        """

        assert self.embedding_model

        # Embed query
        query_vector = self.embedding_model([query_str]).cpu().numpy()

        return self.vector_query(query_vector, num_neighbors)


def data_to_dict(data_path : str):
    """Read Pile data file into dictionary.
    
    Parameters
    ----------
    data_path : str
        Path to Pile data file.
        Assumes json line format with 'text' key.
        
    Returns
    -------
    dict
        Dictionary mapping index to data item.
    """

    print('Reading data file: ', data_path)

    texts = []
    with open(data_path, 'r') as data_file:
        for line in tqdm(data_file):
            texts.append(json.loads(line)['text'])

    return dict(zip(range(len(texts)), texts))


def build_index(data_path : str, index_path : str):
    """Build index from Pile data and index files."""

    index = faiss.read_index(index_path)
    data_dict = data_to_dict(data_path)

    return index, data_dict


def build_roberta_index(data_file : str):
    """Convenience method to build roberta index.
    
    Parameters
    ----------
    data_file : str
        Name of Pile data file.
    
    Returns
    -------
    PileIndex
        Pile index.
    """

    data_path = os.path.join('pile/train', data_file)
    index_path = os.path.join('indexes/roberta-large',
                              data_file + '.index')
    assert os.path.exists(data_path)
    assert os.path.exists(index_path)
    index, data_dict = build_index(data_path, index_path)
    return PileIndex(index, data_dict)


def split_index_data(pile_index : PileIndex, num_splits : int):
    """Split index and dat into num_splits pieces.
    
    Parameters
    ----------
    pile_index : PileIndex
        Index to split.
    num_splits : int
        Number of splits to make.
    
    Yields
    ------
    PileIndex
    """

    index = pile_index.index
    data_dict = pile_index.data_dict

    chunk_size, remainder = divmod(index.ntotal, num_splits)
    for i in range(0, num_splits):
        offset = i * chunk_size
        # handle last chunk
        if i == num_splits - 1:
            chunk_size += remainder
        vectors = index.reconstruct_n(offset, chunk_size)
        data_split = [data_dict[k] for k in range(offset, offset + chunk_size)]
        index_split = faiss.IndexFlatL2(index.d)
        index_split.add(vectors)
        yield PileIndex(index_split, data_split, pile_index.embedding_model)
