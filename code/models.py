import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from advanced_layers import Ranker
import time
import numpy as np
import json
from util import *

class Model(nn.Module):

    def __init__(self, args, emb):
        super(Model, self).__init__()
        self.args = args
        self.emb = emb
        self.ranker = Ranker(args, emb, args.learning_rate, args.dropout)
        self.train_step_enc = torch.optim.Adam(self.ranker.parameters(),
                                          lr = self.ranker.lr)

    def forward(self, bx, by, bs, bo, ans, kb, sub_idx, dic, hop = 3, golden_num = None):
        dic, dic2 = dic
        self.ranker.train()
        vbx = Variable(torch.LongTensor(bx), requires_grad=False).cuda()
        vbs = Variable(torch.LongTensor(bs), requires_grad=False).cuda()
        self.train_step_enc.zero_grad()

        ytables = []
        t, top, stop, threshold = 0, self.args.beam, 0, self.args.threshold
        question = self.ranker.encoder(vbx)
        vbyy = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
        accum_stop = Variable(torch.FloatTensor([0]), requires_grad=True).cuda()
        while t < hop and stop < threshold:
            if t == 0:
                dep_hc = Variable(torch.zeros((question.size()[0], 1, question.size()[2], 1)), requires_grad=False).cuda()
            vlogits, question, vstop, dep_hc = self.ranker.ranker(question, vbs, dep_hc)
            stop = vstop.cpu().data.numpy()

            if (golden_num and t == (golden_num - 1)) or (golden_num is None and np.max(by[0, :]) == 1.):
                vbyy[0] = 1
            temp_stop = -(1.*vbyy*torch.log(torch.clamp(vstop, min = 1e-10))+
                (1.-vbyy)*torch.log(torch.clamp(1-vstop, min = 1e-10)))
            accum_stop = accum_stop + temp_stop
            print('target stop %s\tstop %s\tloss%s' %(vbyy.cpu().data.numpy(), stop, temp_stop.cpu().data.numpy()))
            if t > 0:
                vbss = Variable(torch.LongTensor(bs), requires_grad=False).cuda()
                vlogits = self.ranker.final_probs(vlogits, vprev_logits, vbss)
            else:
                vlogits = torch.squeeze(vlogits, 1)

            if t+1 < hop and stop < threshold:
                logits = vlogits.cpu().data.numpy()
                by, nbs, bo, ytable, top_logits, bs = obtain_next_xyz(logits,
                    bx, bs, bo, ans, kb, sub_idx, [dic, dic2], is_prev_log = True, top=top)
                vbs = Variable(torch.LongTensor(nbs), requires_grad=False).cuda()
                vlogits_mask = Variable(torch.ByteTensor(top_logits), requires_grad=False).cuda()
                vprev_logits = torch.torch.masked_select(vlogits, vlogits_mask).view(-1, np.min([top_logits.shape[1], top]))
                B, _, QL, n_d = question.size()
                vlogits_mask = torch.unsqueeze(torch.unsqueeze(vlogits_mask, -1), -1)
                question = torch.torch.masked_select(question, vlogits_mask).view(B, np.min([top_logits.shape[1], top]), QL, -1)
                dep_hc = torch.torch.masked_select(dep_hc, vlogits_mask).view(B, np.min([top_logits.shape[1], top]), QL, 1)
            t += 1

        by /= np.expand_dims(np.maximum(np.sum(by, 1), 1.e-10), 1)
        vby = Variable(torch.FloatTensor(by), requires_grad=False).cuda()
        cost = self.ranker.obtain_reward(vby, vlogits, accum_stop)

        time1 = time.time()
        cost.backward(retain_graph = False)
        self.train_step_enc.step()
        cost = self.ranker.loss.cpu().data.numpy()

        preds = vlogits.cpu().data.numpy()
        loss = cost
        return cost, loss, cost, preds, t, bs, bo, by

    def check_accuracy(self, bx, by, bs, bo, ans, kb, sub_idx, dic, hop = 3):
        dic, dic2 = dic
        self.ranker.eval()
        vbx = Variable(torch.LongTensor(bx), requires_grad=False).cuda()
        vbs = Variable(torch.LongTensor(bs), requires_grad=False).cuda()

        bstables = []
        botables = []
        bytables = []
        predtables = []
        t, top, stop, threshold = 0, self.args.beam, 0, self.args.threshold

        question = self.ranker.encoder(vbx) # encode questions
        while t < hop and stop < threshold:
            time1 = time.time()
            bstables += [bs]
            botables += [bo]
            bytables += [by]

            if t == 0:
                dep_hc = Variable(torch.zeros((question.size()[0], 1, question.size()[2], 1)), requires_grad=False).cuda()
            # rank the relations in this iteration
            vlogits, question, vstop, dep_hc = self.ranker.ranker(question, vbs, dep_hc)
            stop = vstop.cpu().data.numpy()

            if t > 0:
                # generate $s^{(t)}$
                vbss = Variable(torch.LongTensor(bs), requires_grad=False).cuda()
                vlogits = self.ranker.final_probs(vlogits, vprev_logits, vbss)
            else:
                vlogits = torch.squeeze(vlogits, 1)

            if t+1 < hop and stop < threshold: # if t is smaller than the maximum step and $\bar{z}^{(t)}$ smaller than the threshold
                logits = vlogits.cpu().data.numpy()
                # generate relations for next iteration
                by, nbs, bo, ytable, top_logits, bs = obtain_next_xyz(logits,
                    bx, bs, bo, ans, kb, sub_idx, [dic, dic2], is_prev_log = True, top=top)
                predtables += [logits]
                vbs = Variable(torch.LongTensor(nbs), requires_grad=False).cuda()
                vlogits_mask = Variable(torch.ByteTensor(top_logits), requires_grad=False).cuda()
                vprev_logits = torch.torch.masked_select(vlogits, vlogits_mask).view(-1, np.min([top_logits.shape[1], top]))
                B, _, QL, n_d = question.size()
                vlogits_mask = torch.unsqueeze(torch.unsqueeze(vlogits_mask, -1), -1)
                question = torch.torch.masked_select(question, vlogits_mask).view(B, np.min([top_logits.shape[1], top]), QL, -1)
                dep_hc = torch.torch.masked_select(dep_hc, vlogits_mask).view(B, np.min([top_logits.shape[1], top]), QL, 1)
            t += 1

        preds = vlogits.cpu().data.numpy()
        predtables += [preds]
        return predtables, bstables, botables, bytables,

class Train_loop(object):

    def __init__(self, args, emb):
        self.args = args
        self.emb = emb
        self.model = Model(args = args, emb = emb).cuda()
        self.save_para = self.model.state_dict().keys()

    def train(self, train, dev, test, kb, sub_idx, dic, outfile, rel = None):
        args = self.args
        dropout = args.dropout
        padding_id = 0
        dic2 = {} # Reverse the dictionary
        for d in dic:
            dic2[dic[d]] = d

        min_val_err, min_test_err, hop = -1, -1, self.args.max_hop

        '''load pre-trained model'''
        if args.load_id:
            try:
                model_dic = torch.load('%s/model%s.ckpt' %(outfile, args.load_id))
                self.model.load_state_dict(model_dic, strict = False)
                print('successfully load pre-trained parameters ...')
            except:
                print('fail to load pre-trained parameters ...')

        for ep in range(args.max_epochs):#args.max_epochs
            processed = 0
            train_cost = 0.
            train_loss = 0.
            train_zdiff = 0.
            train_err = 0.
            train_preds = []
            train_eval = []
            train_bo_len = []

            if args.is_train:
                train_shuffle_batch = create_batches(len(train[0]), args.batch)#[:5]
            N = len(train_shuffle_batch) if args.is_train else 0

            for i in range(N): # by, bs, bo, ba, kb
                bx, by, bs, bo, ba = obtain_xys(train, kb, sub_idx, train_shuffle_batch[i], [dic, dic2])
                print('>>> batch: %s\ttask %s\tsave id %s\tbx %s\tby %s\tbs %s'
                    %(i, args.task, args.save_id, str(bx.shape), str(by.shape), str(bs.shape)))

                golden_num = rel[0][0][train_shuffle_batch[i][0]] if ('PathQuestions' in outfile) else None # or 'MetaQA' in outfile
                cost, loss, zdiff, probs, display, bs, bo, by = self.model.forward(bx,
                        by, bs, bo, ba, kb, sub_idx, [dic, dic2], hop =hop, golden_num = golden_num)
                err, pred = get_F1(probs, bs, train[1], bo, dic2, train_shuffle_batch[i], rel[0][1])
                print('##############')
                print(pred[0][1])
                print(rel[0][1][train_shuffle_batch[i][0]])

                k = len(by)
                processed += k
                train_cost += cost
                train_loss += loss
                train_zdiff += np.sum(zdiff)
                train_err += np.sum(err)
                train_preds += pred
                train_eval += err

                for b in range(k):
                    print('index %s\tgenerate num %s\teval %s\ttarget %s\t%s-%s'
                        %(train_shuffle_batch[i][b], len(bo[b]), err[b], np.sum(by[b]),
                        display, rel[0][0][train_shuffle_batch[i][b]]))
                    train_bo_len += [len(bo[b])]

                # if (i+1) % 10 == 0:
                #     embedding = self.model.ranker.emb_init.weight.cpu().data.numpy()
                #     np.save('%s/weights%s' %(outfile, args.save_id), embedding)
                #     saver = get_weights_and_biases(self.model, self.save_para)
                #     torch.save(saver, '%s/model%s.ckpt' %(outfile, args.save_id))

            train_cost /= np.max([processed, 1.e-10])
            train_loss /= np.max([processed, 1.e-10])
            train_err /= np.max([processed, 1.e-10])
            train_zdiff /= np.max([processed, 1.e-10])

            message = '%d | train loss %g(%g)(%g) | train eval: %g ' \
                    %(ep+1, train_cost, train_loss, train_zdiff, train_err)

            print('Evaluation ... ')

            if args.only_eval:
                shuffle_batch = [[6740]]
            else:
                shuffle_batch = create_batches(len(dev[0]), args.batch)

            N = len(shuffle_batch)
            valid_err = 0.
            processed = 0
            valid_eval = []
            valid_preds = []
            valid_bo_len = []
            for i in range(N): #x, y, s, o, a
                bx, by, bs, bo, ba = obtain_xys(dev, kb, sub_idx, shuffle_batch[i], [dic, dic2], is_train=False)
                # print('>>> batch: %s\tbx %s\tby %s\tbs %s' %(i, bx.shape,
                #     by.shape, bs.shape))

                probs, bs, bo, by = self.model.check_accuracy(bx,
                        by, bs, bo, ba, kb, sub_idx, [dic, dic2], hop =hop)
                err, pred = get_F1(probs[-1], bs[-1], dev[1], bo[-1], dic2, shuffle_batch[i], rel[1][1])

                k = len(by[-1])
                processed += k
                valid_err += np.sum(err)
                valid_eval += err
                valid_preds += pred

                for b in range(k):
                    if args.only_eval:
                        print('\n>> Q: %s\t>>> A:%s' %(idx2word(bx[b, :], dic2),
                            dev[1][shuffle_batch[i][b]]))
                        for t in range(len(by)):
                            for j in range(bs[t].shape[1]):
                                if bs[t][b, j, 0] != 0:
                                    try:
                                        prob = probs[t][b, j]
                                    except:
                                        prob = ''
                                    print('hop %s\t%s\t%s\t%s' %(t,
                                        idx2word(bs[t][b, j, :], dic2),
                                        by[t][b, j], prob))
                                    if j < len(bo[t][b]):
                                        print('--->%s'%str([idx2word(w, dic2) for
                                            w in bo[t][b][j][:10]]))
                    print('batch %s\tindex %s\tgenerate num %s\teval %s\ttarget %s'
                        %(i, shuffle_batch[i][b], len(bo[-1][b]), err[b], np.sum(by[-1][b])))
                    valid_bo_len += [len(bo[-1][b])]

            valid_err /= np.max([processed, 1.e-10])
            message += ' | val eval: %g ' %valid_err
            print(message)

            test_err = 0.
            if valid_err > min_val_err:
                shuffle_batch = create_batches(len(test[0]), args.batch)

                processed = 0
                test_eval = []
                test_preds = []
                test_bo_len = []
                N = len(shuffle_batch)
                for i in range(N):
                    bx, by, bs, bo, ba = obtain_xys(test, kb, sub_idx, shuffle_batch[i], [dic, dic2], is_train=False)

                    probs, bs, bo, by = self.model.check_accuracy(bx,
                        by, bs, bo, ba, kb, sub_idx, [dic, dic2], hop =hop)
                    err, pred = get_F1(probs[-1], bs[-1], test[1], bo[-1], dic2, shuffle_batch[i], rel[2][1])

                    k = len(by[-1])
                    processed += k
                    test_err += np.sum(err)
                    test_eval += err
                    test_preds += pred

                    for b in range(k):
                        # print('batch %s\tindex %s\tgenerate num %s\teval %s\ttarget %s'
                        #     %(i, shuffle_batch[i][b], len(bo[-1][b]), err[b],
                        #         np.sum(by[-1][b])))
                        test_bo_len += [len(bo[-1][b])]

                test_err /= np.max([processed, 1.e-10])
                message += ' | test eval: %g ' %test_err

                if test_err > min_test_err:
                    min_test_err = test_err
                    test_preds = print_pred(test_preds, shuffle_batch, test_eval)
                    save_pred('%s/pred%s_test.txt' %(outfile, args.save_id), test_preds)
                    #test_preds = print_pred(test_bo_len, shuffle_batch)
                    #save_pred('%s/pred%s_test_bolen.txt' %(outfile, args.save_id), test_preds)

            print(message)
            if args.is_train:
                if (test_err == 0 and valid_err > min_val_err) or (test_err > 0 and test_err == min_test_err):
                    min_val_err = valid_err
                    embedding = self.model.ranker.emb_init.weight.cpu().data.numpy()
                    np.save('%s/weights%s' %(outfile, args.save_id), embedding)
                    saver = get_weights_and_biases(self.model, self.save_para)
                    torch.save(saver, '%s/model%s.ckpt' %(outfile, args.save_id))
                    message += ' (saved model)'
            else:
                exit()

            log = open('%s/result%s.txt' %(outfile, args.save_id), 'a')
            log.write(message + "\n")
            log.close()

def get_weights_and_biases(model, save_para):
    state_dict = {}
    old_dict = model.state_dict()
    for var in old_dict:
        if 'emb_init' not in var and var in save_para:
            state_dict[var] = old_dict[var]
    return state_dict
