# A Streaming Approach For Efficient Batched Beam Search

This repo contains an implementation of the main machine translation experiments for the EMNLP 2020 paper "A Streaming Approach For Efficient Batched Beam Search" by Kevin Yang, Violet Yao, John DeNero, and Dan Klein, (TODO link). For the code for semantic and syntactic parsing experiments see https://github.com/yangkevin2/emnlp2020-stream-beam-semantic and https://github.com/yangkevin2/emnlp2020-stream-beam-syntactic. 

Main implementation of variable-width beam searches is in `variable_stream.py`. 

## Setup:

We use Fairseq's MT checkpoints. test.py loads and caches them to run inference. 

Requirements: fairseq 0.9.0 and dependencies. 
Important note: Make sure to clone fairseq and install using `pip install -e`. Then in your fairseq install, modify the line `transformer.py:709` to the following: 

`self.embed_positions.weights[prev_output_tokens.shape[1] + 1 - (prev_output_tokens.cumsum(dim=1) == 0).sum(dim=1)].unsqueeze(1)`

The non-variable-width baselines don't need this change. 

Download the newstest2017 and newstest2018 sgm data files from http://www.statmt.org/wmt19/translation-task.html, converting to sgm using the provided scripts if needed. 

## Example command to run after finishing setup:

`python test.py --data_dir {{PATH_TO_DATA_DIR}} --method variable_stream --model ensemble --device cuda --max_tokens 1000 --limit 100000000 --k 50 --ap 1.5 --mc 5 --eps 0.167 --srclang de --trglang en --data_split dev`

`--data_dir` should point to your directory containing the newstest2017 and newstest2018 sgm files. Switch `--data_split dev` to `--data_split test` for the test set (newstest2018). In the args, `--k` is beam size, `--ap` is delta, `--mc` is M, `--eps` is epsilon in the paper. `--max_tokens` is the source-side max tokens per batch. `--limit` is the max number of sentences to translate (set to a large number to do the entire set).

If you need to get number of candidate expansions for the Fixed and Greedy baselines, see the notes for modifying the code in `test.py`. 



