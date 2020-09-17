import torch
from bs4 import BeautifulSoup
import sacrebleu
import copy
from typing import *
import argparse
import time
import os

from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import data_utils, FairseqDataset, iterators, Dictionary

from sequence_generator import * # NOTE: import from old_sequence_generator instead when testing greedy/beam baselines to get statistics

def sgm_to_lst(f):
    f = open(f, 'r')
    data= f.read()
    soup = BeautifulSoup(data)
    return [t.getText() for t in soup.findAll('seg')]

# get sacreBLEU
def evaluate_newstest(model, src, trg, beam=5):
    """
    fairseq evaluation code: https://github.com/pytorch/fairseq/blob/master/fairseq_cli/score.py#L54
    if args.sacrebleu:
        import sacrebleu

        def score(fdsys):
            with open(args.ref) as fdref:
                print(sacrebleu.corpus_bleu(fdsys, [fdref]))
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        predictions = model.translate(src, beam = beam)
    print(predictions)
    return sacrebleu.corpus_bleu(predictions, [trg]).score

def custom_eval(model, src, trg, beam=5, ap=math.inf, eps=1./6, mc=None, method=None):
    model.eval()
    with torch.no_grad():
        tokenized_sentences = [model.encode((sentence)) for sentence in src]
        gen_args = copy.copy(model.args)
        gen_args.beam = beam
        gen_args.mc = mc
        generator = build_generator(model.task, model.models, gen_args)
        results = []
        # model.args.max_sentences = 64
        total_loops, total_expansions = 0, 0
        if method == 'variable_stream':
            # TODO adjust other parameters; adjust batching params
            ids, translations, total_loops, total_expansions = generator.variable_beam_stream(model, tokenized_sentences, bos_token=model.task.target_dictionary.eos(), ap=ap, mc=mc, eps=eps)
            for id, hypos in zip(ids, translations):
                results.append((id, hypos))
        else:
            for batch in model._build_batches(tokenized_sentences, False):
                # print('b')
                batch = utils.apply_to_sample(lambda t: t.to(model.device), batch)
                if method is None:
                    translations, n_loops, n_expansions = generator.generate(model.models, batch, bos_token=model.task.target_dictionary.eos(), ap=ap)
                elif method == 'greedy':
                    translations, n_loops, n_expansions = generator.greedy(model.models, batch, bos_token=model.task.target_dictionary.eos())
                elif method == 'variable_beam':
                    translations, n_loops, n_expansions = generator.variable_beam(model, batch, bos_token=model.task.target_dictionary.eos(), ap=ap, mc=mc)
                total_loops += n_loops
                total_expansions += n_expansions
                for id, hypos in zip(batch["id"].tolist(), translations):
                    results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        predictions = [model.decode(hypos[0]['tokens']) for hypos in outputs]
        bleu = sacrebleu.corpus_bleu(predictions, [trg]).score
        # print(predictions)
        print('loops', total_loops)
        print('expansions', total_expansions)
        print(bleu)
        return bleu

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='single')
    parser.add_argument('--srclang', type=str, default='de')
    parser.add_argument('--trglang', type=str, default='en')
    parser.add_argument('--method', type=str)
    parser.add_argument('--data_split', type=str, default='dev')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--ap', type=float, default=1.5)
    parser.add_argument('--mc', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.167)
    parser.add_argument('--max_tokens', type=int, default=3584)
    parser.add_argument('--limit', type=int, default=100000000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    assert 0 < args.eps < 1, 'Epsilon should be between 0 and 1'

    if args.model == 'single':
        translation_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.' + args.srclang + '-' + args.trglang + '.single_model', tokenizer='moses', bpe='fastbpe')
    else:
        assert args.model == 'ensemble'
        translation_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.' + args.srclang + '-' + args.trglang, checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                        tokenizer='moses', bpe='fastbpe')
    if 'cuda' in args.device:
        translation_model = translation_model.to(args.device)
    translation_model.args.max_tokens = args.max_tokens

    if args.data_split == 'dev':
        src_full = sgm_to_lst(os.path.join(args.data_dir, "newstest2017-" + args.srclang + args.trglang + "-src." + args.srclang + ".sgm"))
        trg_full = sgm_to_lst(os.path.join(args.data_dir, "newstest2017-" + args.srclang + args.trglang + "-ref." + args.trglang + ".sgm"))
    else:
        assert args.data_split == 'test'
        src_full = sgm_to_lst(os.path.join(args.data_dir, "newstest2018-" + args.srclang + args.trglang + "-src." + args.srclang + ".sgm"))
        trg_full = sgm_to_lst(os.path.join(args.data_dir, "newstest2018-" + args.srclang + args.trglang + "-ref." + args.trglang + ".sgm"))

    print('length of src', len(src_full[:args.limit]))

    src_debug = ["Maschinelles Lernen ist großartig!"]
    trg_debug = ["Machine learning is great!"]

    print('starting')
    print(time.ctime())
    start = time.perf_counter()
    if args.method == 'greedy':
        # NOTE: switch to the commented line and use old_sequence_generator instead of sequence_generator to display loop/expansion statistics
        bleu = evaluate_newstest(translation_model, src_full[:args.limit], trg_full[:args.limit], beam=1)
        # bleu = custom_eval(translation_model, src_full[:args.limit], trg_full[:args.limit], beam=1, ap=1e8, mc=1, method=None)
    elif args.method == 'beam':
        # NOTE: switch to the commented line and use old_sequence_generator instead of sequence_generator to display loop/expansion statistics
        bleu = evaluate_newstest(translation_model, src_full[:args.limit], trg_full[:args.limit], beam=args.k)
        # bleu = custom_eval(translation_model, src_full[:args.limit], trg_full[:args.limit], beam=args.k, ap=1e8, mc=args.k, method=None)
    elif args.method == 'variable_beam':
        bleu = custom_eval(translation_model, src_full[:args.limit], trg_full[:args.limit], beam=args.k, ap=args.ap, mc=args.mc, method='variable_beam')
    else:
        assert args.method == 'variable_stream'
        bleu = custom_eval(translation_model, src_full[:args.limit], trg_full[:args.limit], beam=args.k, ap=args.ap, mc=args.mc, eps=args.eps, method='variable_stream')
    end = time.perf_counter()

    print(time.ctime())
    print('time in seconds', end - start)
    if 'cuda' in args.device:
        print('max GB gpu mem used', torch.cuda.max_memory_allocated(device=translation_model.device) / 2**30) # max mem used until now, in GB
    print('bleu', bleu)
