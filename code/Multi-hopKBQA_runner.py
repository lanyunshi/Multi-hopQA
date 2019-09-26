from options import load_arguments
from util import *
import numpy as np
from time import gmtime, strftime
import os
import pickle
import re

from models import Train_loop

def main(args, load_save = None):
    print("Train and test for task  ..." )
    if load_save:
        outfile = load_save
        args = pickle.load(open('%s/config.pkl' %outfile, 'rb'))
        dic = pickle.load(open('%s/dic.pkl' %outfile, 'rb'), encoding='latin1') #
    else:
        outfile = 'trained_model/' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        dic = {"<unk>": 0, "-1": 1}

    '''Change or not change the saved arguments'''
    # args.task = load_save
    # args.load_id = '_Acc'
    # args.save_id = '_Acc'
    args.is_train = 1
    args.only_eval = 0

    folder = args.train_q.split('/')[-2]
    if load_save:
        '''
        If data is processed, load the pre-processed data, kb and embedding.
        '''
        # load questions and answers
        train_x, train_y, train_e = read_annotations(args.train_q, dic)
        dev_x, dev_y, dev_e = read_annotations(args.dev_q, dic)
        test_x, test_y, test_e = read_annotations(args.test_q, dic)
        # load the kb and pre-trained embeddings
        kb, sub_idx = pickle.load(open('%s/kb.pkl' %outfile, 'rb'))
        if os.path.isfile('%s/weights%s.npy' %(outfile, args.load_id)):
            emb = np.array(np.load('%s/weights%s.npy' %(outfile, args.load_id)))
        else:
            emb = initialize_vocab(dic, args.embedding)
            np.save("%s/weights%s" %(outfile, args.save_id), emb)
    else:
        # preprocess data
        train_x, train_y, train_e = read_annotations(args.train_q, dic)
        dev_x, dev_y, dev_e = read_annotations(args.dev_q, dic)
        test_x, test_y, test_e = read_annotations(args.test_q, dic)
        kb, sub_idx = read_kb(args.kb_file, dic)
        emb = initialize_vocab(dic, args.embedding)

        pickle.dump([train_x, train_y, train_e, dev_x, dev_y, dev_e,
            test_x, test_y, test_e],
            open('%s/q.pkl' %outfile, 'wb'), protocol = 2)
        pickle.dump([kb, sub_idx], open('%s/kb.pkl' %outfile, 'wb'), protocol = 2)

        pickle.dump(dic, open('%s/dic.pkl' %outfile, 'wb'), protocol = 2)
        np.save("%s/weights%s" %(outfile, args.save_id), emb)

    print('train: %s dev: %s test: %s' %(len(train_x), len(dev_x), len(test_x)))

    train_r = read_golden_rel('data/mix-hop/%s/qa_train_qtype.txt' %folder)
    dev_r = read_golden_rel('data/mix-hop/%s/qa_dev_qtype.txt' %folder)
    test_r = read_golden_rel('data/mix-hop/%s/qa_test_qtype.txt' %folder)
    rel = (train_r, dev_r, test_r)

    print('Parser Arguments')
    for key, value in args.__dict__.items():
        print(u'{0}: {1}'.format(key, value))

    train_loop = Train_loop(args = args, emb = emb)
    train_loop.train((train_x, train_y, train_e),
                    (dev_x, dev_y, dev_e),
                    (test_x, test_y, test_e),
                    kb, sub_idx, dic, outfile, rel = rel)

if __name__ == '__main__':
    args = load_arguments()

    if args.task == 1: # x-hop PathQuestions
        main(args, load_save = 'trained_model/PathQuestions')
    elif args.task == 2: # x-hop WC2014
        main(args, load_save = 'trained_model/WC2014')
    elif args.task == 3: # x-hop MetaQA
        main(args, load_save = 'trained_model/MetaQA')
    elif args.task == 0:
        main(args)

'''
CUDA_VISIBLE_DEVICES="5" python code/Multi-hopKBQA_runner.py --task 3
'''
