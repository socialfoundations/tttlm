# Test-Time Training on Nearest Neighbors (TTT-NN)

The repository contains code to:

1. Train the embedding model
2. Build nearest neighbor index on the Pile
3. Run distributed servers on top of the index
4. Query the servers
5. Evaluate TTT-NN on the Pile
6. Run baselines

This project is based on the paper [Test-Time Training on Nearest Neighbors for Large Language Models](https://arxiv.org/abs/2305.18466) by Moritz Hardt and Yu Sun, in ICLR 2024. Please cite as:

```
@inproceedings{hardt2024test,
  title={Test-time training on nearest neighbors for large language models},
  author={Hardt, Moritz and Sun, Yu},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Necessary files

To evaluate TTT-NN you ultimately need the following directory structure:

```
indexes/
  roberta-large/
    00.jsonl.index
    01.jsonl.index
    ...
    29.jsonl.index

models/
  roberta-large-pile-lr2e-5-bs16-8gpu/
    checkpoint-1700000/

pile/
  train/
    00.jsonl
    01.jsonl
    ...
    29.jsonl
  val.jsonl
  test.jsonl

servers/
  addresses.txt
```

Download the dataset [here](https://the-eye.eu/public/AI/pile/) and place the files in the `pile/` subdirectory.

## Train or download the embedding model

### Download the embedding model

You can download the pretrained embedding model from [HuggingFace](https://huggingface.co/socialfoundations/roberta-large-pile-lr2e-5-bs16-8gpu-1700000). Place the files in the directory `models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000`.

### Alternatively, train the embedding model

To train the embedding model yourself, see `code/trainer_lm.py`. This is a standard HuggingFace training setup.
This code was used to produce the model checkpoint `checkpoint-1700000` in the `models` directory.
The model trained for approximately one month on 8 A100 GPUs, making one pass over the data.

Make sure to have the checkpoint `models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000` before you proceed.

## Build or download the index

### Download the index

You can download the index files [here](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.EJQGAK). The download size is approximately 800GB. Place the files in the directory `indexes/roberta-large`.

### Alternatively, build the index

To build the index yourself, use the code in `code/build_database.py` to build an index on top of the Pile dataset. This is a time consuming operation.  Specify `--data_file` to build index for given data file.  

```
python3 code/build_database.py \
        --data_file pile/train/00.jsonl \
        --output_dir indexes/roberta-large
```

Make sure you have all index files in `indexes/roberta-large` before you proceed.

## Run the Pile server

The following command will launch a server with 6 replicas each serving one split of the data. This will append 6 ip addresses and ports to the file specified as `address_path`. 

```
python3 code/pile_server.py \
        --address_path servers/addresses.txt \
        --data_file pile/train/00.jsonl \
        --num_servers 6
```

To serve from all Pile data files, start one server for each data file. 
We recommend starting 30 servers with 6 replicas each, resulting in 180 instances running.

Make sure servers are up and running before launching evaluation.

## Use the Pile client

Use `code/pile_client.py` to query the server. Specify `--address_path` to indicate which servers to query. The client will query all servers it finds under the address path and query each. The client then builds a local nearest neighbors structure to find the nearest neighbors among all the retrieved results.

The client code can be used as a standalone client, but will also be called from the evaluation code.

## Run test-time training with nearest neighbors

To evaluate on GPTNeo with default parameters:

```
python3 code/eval_tttlm.py \
        --address_path servers/addresses.txt \
        --results_dir results/
```

To evaluate on GPT2:

```
python3 code/eval_tttlm.py \
        --model gpt2-large \
        --tokenizer gpt2-large \
        --embedding_model_checkpoint models/roberta-large-pile-lr2e-5-bs16-8gpu/checkpoint-1700000
        --max_length 1024 \
        --stride 1024 \
        --learning_rate 2e-5 \
        --address_path servers/addresses.txt \
        --results_dir results/
```

Replace `gpt2` with `gpt2-large` to evaluate on GPT2Large.

Use `code/process_results.py` to merge results for distributed evaluation with multiple processes, aggregate statistics, and make plots.

The evaluation code uses modified parts of the [`lm_evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) package by Eleuther-AI. 
The folder ```lm_eval``` contains the modified parts (also ```lm_eval_interpolation``` for the interpolation baseline), as well as the unmodified parts to ensure compatibility.

## Run baselines

* See `code/baseline_context.py` for in-context baseline. Also `code/process_results_context.py`.
* See `code/baseline_interpolation.py` for interpolation baseline. Also `code/process_results_interpolation.py`.
* Run `code/eval_tttlm.py` with option `--dynamic_eval` for dynamic evaluation baseline.
