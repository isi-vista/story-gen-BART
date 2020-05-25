import argparse
import os
import sys
import numpy as np

import torch
from fairseq.models.bart import BARTModel


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--data_type', type=str, choices=["plot", "story"])
    p.add_argument('--ranking', action="store_true", help="output ranking accuracy of x vs y pairs")
    p.add_argument('--gen_data', nargs="+", help="generated files")
    p.add_argument('--cond_data', type=str, help="filename of conditionally generated data")
    p.add_argument('--gold', type=str, help="gold stories, necessary for ranking accuracy")
    return p.parse_args()



if __name__ == "__main__":
    args = setup_argparse()
    # prints results so need to redirect
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    use_cuda = torch.cuda.is_available()

    if args.data_type == "plot":
        ### load BART model
        bart = BARTModel.from_pretrained(
            'checkpoint_full/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='full'
        )
    elif args.data_type == "story":
        bart = BARTModel.from_pretrained(
            'checkpoint-fullstory/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='fullstory'
        )

    bart.eval()
    if use_cuda:
        bart.cuda()  # remove this line if not running with cuda
        bart.half()  # doesn't work with CPU

    if args.ranking:
        with open(args.gold, "r") as fin:
            gold_stories = fin.readlines()

    with open(args.cond_data, "r") as fin:
        cond_data = fin.readlines()

    for file in args.gen_data:
        print("Working on {}...".format(file))
        # run sequence scorer
        gold_win, gen_win = 0, 0
        all_scores = []
        with open(file, "r") as fin:
            for i, line in enumerate(fin):
                with torch.no_grad():
                    score = bart.score_sequence([line.strip()], [cond_data[i].strip()])
                    #print(score)
                    all_scores.append(score)
                    if args.ranking:
                        gold_score = bart.score_sequence([gold_stories[i].strip()], [cond_data[i].strip()])
                        if gold_score > score:
                            gold_win += 1
                        else:
                            gen_win += 1

        if args.ranking:
            print("Generated ranked above gold: {:.2f}".format(gen_win/(gold_win+gen_win)))

        else:
            print("Mean conditional probability of sequences: {:.2f}".format(np.mean(all_scores)))

