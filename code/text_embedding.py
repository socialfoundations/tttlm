"""Text embeddings from various language models."""

from abc import ABC, abstractmethod

import torch

from typing import List

from transformers import RobertaTokenizer, RobertaModel
from transformers import GPT2Tokenizer, GPTNeoModel


class TextEmbedding(ABC):
    """Text embedding base class."""

    @property
    @abstractmethod
    def device(self):
        """Device to use for embedding."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self):
        """Dimension of the embedding."""
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        """Dimension of the embedding."""
        pass

    @property
    @abstractmethod
    def model(self):
        """Return embedding model."""
        pass

    def __call__(self, texts: List[str]):
        """Get text embeddings for a list of strings.
        
        Parameters
        ----------
        texts : List[str]
            List of strings to embed.
            
        Returns
        -------
        emb : torch.Tensor of size ([len(texts), embedding_dimension])
        
        See Also
        --------
        Source: https://nn.labml.ai/transformers/retro/bert_embeddings.html
        """

        with torch.no_grad():

            tokens = self.tokenizer(texts, truncation=True, return_tensors='pt',
                                    padding=True, add_special_tokens=True)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask)

            # Get the embedding layer
            state = output['last_hidden_state']

            # Calculate the average token embeddings.
            # Note that the attention mask is `0` if the token is empty padded.
            # We get empty tokens because the texts are of different lengths.
            emb = (state * attention_mask[:, :, None]).sum(dim=1) 
            emb /= attention_mask[:, :, None].sum(dim=1)

            return emb


class GPTNeoEmbedding(TextEmbedding):
    """GPTNeo text embeddings."""

    def __init__(self, device: torch.device = torch.device('cuda')):
        """Initialize GPTNeoEmbedding."""
        model_id = 'EleutherAI/gpt-neo-1.3B'
        self._device = device
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self._tokenizer.pad_token = self.tokenizer.eos_token
        self._model = GPTNeoModel.from_pretrained(model_id)
        self._model.to(device)

    @property
    def device(self):
        """Device to use for embedding."""
        return self._device

    @property
    def embedding_dimension(self):
        """Dimension of the embedding."""
        # EleutherAI/gpt-neo-1.3B has d=2048,
        # EleutherAI/gpt-neo-2.7B has d=2560
        return 2048

    @property
    def tokenizer(self):
        """Tokenizer for the embedding."""
        return self._tokenizer

    @property
    def model(self):
        """Return embedding model."""
        return self._model


class RobertaEmbedding(TextEmbedding):
    """Roberta text embeddings."""

    def __init__(self, embedding_model: str = 'roberta-large',
                       device: torch.device = torch.device('cuda')):
        """Initialize RobertaEmbedding."""
        self._device = device
        self._tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self._model = RobertaModel.from_pretrained(embedding_model)
        self._embedding_dimension = self._model.config.hidden_size
        self._model.to(device)

    @property
    def device(self):
        """Device to use for embedding."""
        return self._device

    @property
    def embedding_dimension(self):
        """Dimension of the embedding."""
        return self._embedding_dimension

    @property
    def tokenizer(self):
        """Tokenizer for the embedding."""
        return self._tokenizer

    @property
    def model(self):
        """Return embedding model."""
        return self._model


def _test_gptneo_embedding():

    gptneo = GPTNeoEmbedding(device=torch.device('cuda'))
    assert hasattr(gptneo, 'embedding_dimension')

    embedding = gptneo(['Hello world!'])
    assert embedding.shape == torch.Size([1, gptneo.embedding_dimension])

    embedding = gptneo(['Hello world!', 'How are you?'])
    assert embedding.shape == torch.Size([2, gptneo.embedding_dimension])

    
def _test_roberta_embedding():

    # Initialize
    roberta = RobertaEmbedding(device=torch.device('cuda'))
    assert hasattr(roberta, 'embedding_dimension')

    embedding = roberta(['Hello world!'])
    assert embedding.shape == torch.Size([1, roberta.embedding_dimension])

    embedding = roberta(['Hello world!', 'How are you?'])
    assert embedding.shape == torch.Size([2, roberta.embedding_dimension])


if __name__ == '__main__':
    _test_roberta_embedding()
    _test_gptneo_embedding()
