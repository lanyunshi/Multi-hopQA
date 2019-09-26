#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
options.py

"""

import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--task",
            type = int,
            default = 1,
        )
    argparser.add_argument("--embedding",
            type = str,
            default = "/home/yunshi/Word2vec/glove.840B.300d.zip",
            help = "path to pre-trained word vectors"
        )
    argparser.add_argument("--save_id",
            type = str,
            default = "_Acc_tmp",
            help = "save index"
        )
    argparser.add_argument("--load_id",
            type = str,
            default = "_Acc_tmp",
            help = "load index"
        )
    argparser.add_argument("--train_q",
            type = str,
            default = "data/mix-hop/WC2014/new_qa_train.txt",
            help = "path to training data"
        )
    argparser.add_argument("--dev_q",
            type = str,
            default = "data/mix-hop/WC2014/new_qa_dev.txt",
            help = "path to development data"
        )
    argparser.add_argument("--test_q",
            type = str,
            default = "data/mix-hop/WC2014/new_qa_test.txt",
            help = "path to test data"
        )
    argparser.add_argument("--kb_file",
            type = str,
            default = "data/mix-hop/WC2014/kb.txt",
            help = "path to kb data"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--batch",
            type = int,
            default = 1,
            help = "mini-batch size"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.0001,# was set to 0.0005
            help = "learning rate"
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.0,
            help = "dropout probability"
        )
    argparser.add_argument("-d", "--hidden_dimension",
            type = int,
            default = 200,
            help = "hidden dimension"
        )
    argparser.add_argument("--is_train",
            type = int,
            default = 1,
            help = "whether train the model or not"
        )
    argparser.add_argument("--only_eval",
            type = int,
            default = 0,
            help = "if it's 1, it's in debug mode. Otherwise, it's evaluation mode."
        )
    argparser.add_argument("--max_hop",
            type = int,
            default = 3,
            help = "Maximum hop number."
        )
    argparser.add_argument("--beam",
            type = int,
            default = 3,
            help = "Beam size."
        )
    argparser.add_argument("--threshold",
            type = int,
            default = 0.5,
            help = "Threshold for termination"
        )
    # added argument for initializer

    args = argparser.parse_args()
    return args

'''
General argument setting

args.load_id = '_Acc_tmp'
args.save_id = '_Acc_tmp'
args.max_epochs = 20
args.batch = 1
args.learning_rate = 0.0001
args.hidden_dimension = 200
args.dropout = 0.
args.is_train = 1 # if is_train=True, train the model; otherwise not train
args.only_eval = 0 # is only_eval=True, display the intermediate results for debugging
'''
