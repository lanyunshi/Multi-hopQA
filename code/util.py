import numpy as np
import sys
import gzip
import random
import json
import re
from collections import defaultdict
import zipfile
import time
import os

np.random.seed(2345)

def read_annotations(path, dic):
    question, answer, entity_pos = [], [], []
    count = defaultdict(int)
    q_len = 0

    with open(path) as f:
        for line_idx, line in enumerate(f):
            q2idx, a2idx = (), ()
            line = line.replace('\n', '').lower()
            line = re.sub("'s", " 's", line)
            line = re.sub("[ ]+", " ", line)
            line = re.sub("\_", " ", line)
            q_txt, a_txt = line.split('\t')

            q_txt = re.sub('(?<=\]),', '', q_txt)
            q_txt = q_txt.strip()
            topic_entity = [not not re.search('[\[\]]', w) for w in q_txt.split(' ')]
            entity_pos += [tuple([i for i, w in enumerate(topic_entity) if w])]
            q_txt = re.sub('[\[\]]', '', q_txt)
            q_txt = re.split(' ', q_txt)
            for i, w in enumerate(q_txt):
                if w not in dic:
                    dic[w] = len(dic)
                q2idx += (dic[w], )
                count[w] += 1

            a_txt = a_txt.split('|')
            for i , w in enumerate(a_txt):
                a2idx += (w, )

            question += [q2idx]
            answer += [a2idx]

    return question, answer, entity_pos

def read_kb(path, dic):
    sub_idx = defaultdict(list)
    kb = defaultdict(list)
    max_line_idx, max_k = 0, 0
    open_tool = open(path)
    with open_tool as f:
        for line_idx, line in enumerate(f):
            line = line.replace('\n', '').lower()
            words = line.split('|')

            if len(words) == 3:
                for i in [0, 1, 2]:
                    sub = ()
                    word = words[i]

                    if i in [0, 2]:
                        word = re.sub("'s", " 's", word)
                        if re.search('\d+/\d+/\d+', word) and not re.search('[a-z]+', word):
                            d, m, y = word.split('/')
                            word = '-'.join([y, m.zfill(2), d.zfill(2)])
                        word = word.split(' ')
                    elif i == 1:
                        #print(word)
                        #word = re.split('\_|[ \.]+', word)
                        word = sum(['_'.join(r.split('.')[-1:]).split('_') for r in re.split('\.\.', word)], [])

                    text = ()
                    for k, w in enumerate(word):
                        if w not in dic:
                            dic[w] = len(dic)
                        text += (dic[w], )
                        sub += (dic[w], )
                    kb[line_idx] += [text]

                    if i == 0:
                        sub_idx[sub] += [(1, line_idx)]
                    elif i == 2:
                        sub_idx[sub] += [(-1, line_idx)]

                    if max_k < k + 1:
                        max_k = k + 1

            if line_idx % 10000000 == 0:
                print('read kb ... %s' %line_idx)

        if max_line_idx < line_idx + 1:
            max_line_idx = line_idx + 1
    return kb, sub_idx

def read_golden_rel(path, is_tranform = False):
    goldens = []
    goldennums = []
    with open(path) as f:
        for line_idx, line in enumerate(f):
            line = line.replace('\n', '').lower()
            golden = []
            line = line.split('|')[0]
            golden_num = len(re.findall('#', line)) if ('MetaQA' in path) else int(len(re.findall('#', line))/2)
            if ('MetaQA' in path):
                golden = re.sub('\#', ' ', line)
            else:
                line = '#'.join([w for i, w in enumerate(line.split('#')) if i not in [2, 4, 6]])
                golden = ' '.join(sum([[w] if re.search('^\$', w) else sum(['_'.join(ww.split('.')[-1:]).split('_') for ww in w.split('..')], []) for w in line.split('#')], []))
            goldens += [golden]
            goldennums += [golden_num]
    return (goldennums, goldens)

def create_batches(N, batch_size, skip_idx = None, is_shuffle = True):
    batches = []
    shuffle_batch = np.arange(N)
    if skip_idx:
        shuffle_batch = list(set(shuffle_batch) - set(skip_idx))
    if is_shuffle:
        np.random.shuffle(shuffle_batch)
    M = int((N-1)/batch_size + 1)
    for i in range(M):
        batches += [shuffle_batch[i*batch_size: (i+1)*batch_size]]
    return batches

def array2tuple(a):
    b = ()
    for _, k in enumerate(a):
        if k == 0:
            break
        else:
            b += (k, )
    return b

def idx2word(a, dic2):
    return ' '.join([dic2[w] for w in a])

def obtain_story(que, kb, sub_idx, dic = None, is_que = True, with_sub = True,
    entity_pos = False, is_train = True):
    '''
    Retrieve the relations based the question or intermediant results
    '''
    def is_sublist(a, b):
        if a in b:
            return True
    def squeeze_story(a, direction, with_sub):
        if direction == 1:
            if with_sub:
                story = a[0] + a[1]
            else:
                story = a[1]
            obj = a[2]
        elif direction == -1: # If the direction is reverse, append an extra symbol
            if with_sub:
                story = a[2] + a[1] + (1, ) # If it's 0-th relation, relation contains (topic entity, relation)
            else:
                story = a[1] + (1, )
            obj = a[0]
        return story, obj

    dic, dic2 = dic
    candidate, prev_obj = [], []
    limit = 10000 if is_train else 1e10
    if is_que: # If the input is the question, string match the topic entities
        for i in range(len(que)):
            for j in range(i+1, len(que)+1):
                if que[i:j] in sub_idx:
                    candidate += sub_idx[que[i:j]]
    else: # Otherwise, starting from the objective of the relation to expand the relations
        for i in range(len(que)):
            if isinstance(que[i][0], int):
                candidate += sub_idx[que[i]]
            else:
                if que[i][0] in sub_idx:
                    candidate += sub_idx[que[i][0]]
                    prev_obj += [que[i][0]]*len(sub_idx[que[i][0]])
                if que[i][1] in sub_idx:
                    candidate += sub_idx[que[i][1]]
                    prev_obj += [que[i][0]]*len(sub_idx[que[i][1]])
        if not isinstance(que[i][0], int):
            if len(candidate) > limit:
                candidate, prev_obj = zip(*random.sample(list(zip(candidate,
                    prev_obj)), limit))

    story, story2idx, idx = [], {}, 0
    mid, obj = defaultdict(list), defaultdict(list)
    #print('%s\t%s' %(len(candidate), set(candidate)))
    for dc in candidate:
        direction, c = dc
        s, o = squeeze_story(kb[c], direction, with_sub)
        if s not in story2idx:
            story += [s]
            story2idx[s] = idx
            idx += 1
        obj[story2idx[s]] += [o]
    obj = [list(set(obj[i])) for i in obj]
    return story, obj

def obtain_xys(data, kb, sub_idx, batches, dic, is_train= True):
    '''
    Obtain initial relations
    Input
    1) data: (question, answer, topic position)
    2) kb: knowledge base
    3) batches: indexes
    4) dic: dictionary from idx to word
    5) is_train: train data or not
    Output
    1) x: question matrix
    2) y: pseudo gold distribution based on the F1 scores of the relation
    3) s: relation in the 0-th iteration
    4) o: object for the relation
    5) a: answer set
    '''
    def remove_topic(obj, bx):
        return [w for w in obj if w not in bx]

    dic, dic2 = dic
    que, ans, entity_pos = data
    x = np.zeros((len(batches), 30), dtype = np.int32)
    e = np.zeros((len(batches), 100), dtype = np.int32)
    y = np.zeros((len(batches), 10000))
    s = np.zeros((len(batches), 10000, 100), dtype = np.int32)
    o, a = [], []

    max_cand, max_xlen, max_slen, max_dlen = 1, 1, 1, 1
    for i, b in enumerate(batches):
        x[i, :len(que[b])] = que[b]
        # print(idx2word(que[b], dic2))
        # print('*****************')
        if max_xlen < len(que[b]):
            max_xlen = len(que[b])

        story, obj = obtain_story(que[b], kb, sub_idx, [dic, dic2],
            entity_pos = entity_pos[b], is_train=is_train)
        for j in range(len(story)):
            slen = np.min([100, len(story[j])])
            s[i, j, :slen] = story[j][:slen]
            if max_slen < len(story[j]):
                max_slen = len(story[j])

            obj[j] = remove_topic(obj[j], que[b]) # remove the relation which has same sub and obj
            temp = naive_get_F1([idx2word(w, dic2) for w in obj[j]], ans[b]) # compute naive F1
            y[i, j] = temp
            # print('%s\t%s\t%s' %(idx2word(story[j], dic2),
            #     [idx2word(w, dic2) for w in obj[j]], ans[b]))

        o += [obj]
        a += [ans[b]]

        if max_cand < len(story):
            max_cand = len(story)

    x = x[:, :max_xlen]
    e = e[:, :max_xlen]
    y = y[:, :max_cand]
    s = s[:, :max_cand, :max_slen]
    #if np.max(y) == 0.: y[:, :] = 0.
    return x, y, s, o, a

def get_F1(probs, bs, bt, bo, dic2, batch, golden_rel, metric='F1'):
    acces, preds = [], []

    for i in range(probs.shape[0]):
        ans, rel = [], []
        top_index = argmax_all(probs[i, :])

        for j in top_index:
            an = bo[i][j] if j < len(bo[i]) else []
            ans += [idx2word(w, dic2) for w in an]
            rel += [idx2word(array2tuple(bs[i, j]), dic2)]

        y_out = ['***'] + list(set(rel)) + ['***'] + ['/'.join(list(set(ans)))]
        preds += [y_out]

        if metric == 'F1':
            #print(ans); print(bt[batch[i]]); exit()
            TP = len(set(ans) & set(bt[batch[i]]))
            precision = TP*1./np.max([len(set(ans)), 1.e-10])
            recall = TP*1./np.max([len(set(bt[batch[i]])), 1.e-10])
            acc = 2*precision*recall/np.max([(precision+recall), 1.e-10])
            acces += [acc]
        elif metric == 'hits1':
            ans = random.sample(set(ans), 1)[0]
            acces += [int(ans in set(bt[batch[i]]))]
        elif metric == 'rel_acc':
            acces += [int(re.sub(' \'', "'", rel[-1]) == re.sub(' \'', "'", golden_rel[batch[i]]))]

    return acces, preds

def argmax_all(l, top_num = 1):
    m = sorted(l)[::-1][:top_num]
    return [i for i,j in enumerate(l) if j in m][:top_num]
    #return np.argsort(l)[::-1][:top_num]

def naive_get_F1(preds, ans):
    precision = len(set(preds)&set(ans))*1./np.max([len(set(preds)), 1e-10])
    recall = len(set(preds)&set(ans))*1./len(set(ans))
    f1 = 2*precision*recall/np.max([precision+recall, 1e-10])
    return f1

def obtain_next_xyz(probs, bx, bs, bo, ans, kb, sub_idx, dic, is_prev_log=False, top=3):
    def remove_sub(obj, bx):
        return [w for w in obj if len(set(w)-set(bx)) != 0]

    dic, dic2 = dic
    y = np.zeros((len(bo), 10000))
    s = np.zeros((len(bo), 10000, 300), dtype = np.int32)
    ytable, m = [], None
    max_cand, max_slen = 1, 1

    prev_logits = np.zeros_like(probs)
    full_s = np.zeros((len(bo), 10000, 300), dtype = np.int32)
    objs, prev_s = {}, {}
    max_j_idx, max_full_slen = 1, 1
    for i in range(probs.shape[0]):
        idx = 0
        seen = {}
        top_index = argmax_all(probs[i, :], top)
        for j_idx, j in enumerate(top_index):
            prev_logits[i, j] = 1
            prev_s[(i, j_idx)] = array2tuple(bs[i, j])
            an = bo[i][j] if j < len(bo[i]) else []

            if len(an) > 0:
                time1 = time.time()
                story, obj = obtain_story(an, kb, sub_idx, [dic, dic2],
                    is_que=False, with_sub = False)

                for k in range(len(story)):
                    #obj[k] = remove_sub(obj[k], bx[i])
                    if prev_s[(i, j_idx)] == array2tuple(story[k]):
                        continue
                    if story[k] not in seen:
                        seen[story[k]] = idx
                        idx += 1
                    s[i, seen[story[k]], :len(story[k])] = story[k]
                    if max_slen < len(story[k]):
                        max_slen = len(story[k])
                    objs[(i, j_idx, seen[story[k]])] = obj[k]
                    ytable += [(i, (seen[story[k]], j))]

        if max_cand < idx:
            max_cand = idx
        if max_j_idx < j_idx + 1:
            max_j_idx = j_idx + 1
    o = [[[]]*(max_cand*max_j_idx) for _ in range(probs.shape[0])]
    for i, j_idx, idx in objs:
        temp = naive_get_F1([idx2word(w, dic2)
                        for w in objs[(i, j_idx, idx)]], ans[i])
        y[i, j_idx*max_cand + idx] = temp
        o[i][j_idx*max_cand + idx] = objs[(i, j_idx, idx)]
        story_unit = prev_s[(i, j_idx)] + array2tuple(s[i, idx])
        full_s[i, j_idx*max_cand + idx, :len(story_unit)] = story_unit
        if max_full_slen < len(story_unit):
            max_full_slen = len(story_unit)

    y = y[:, :max_cand*max_j_idx]
    full_s = full_s[:, :max_cand*max_j_idx, :max_full_slen]
    s = s[:, :max_cand, :max_slen]
    #if np.max(y) == 0.: y[:, :] = 0.
    # for k in range(full_s.shape[1]):
    #     print('%s\t%s\t%s' %(idx2word(full_s[0, k], dic2),
    #         [idx2word(w, dic2) for w in o[0][k][:3]], y[0, k]))
    # exit()
    return y, s, o, ytable, prev_logits, full_s

def initialize_vocab(dic, path):
    vocab = np.random.uniform(-0.1, 0.1, (len(dic), 300))
    seen = 0

    gloves = zipfile.ZipFile(path)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0].decode('utf-8')
                    embedding = splitline[1:]
                    if word in dic and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        vocab[dic[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen += 1

    vocab = vocab.astype(np.float32)
    vocab[0, :]  = 0.
    print("pretrained vocab %s among %s" %(seen, len(dic)))
    return vocab

def print_pred(preds, shuffle_batch, evals = None):
    shuffle_batch = np.concatenate(shuffle_batch)
    idx = sorted(range(len(shuffle_batch)), key = lambda x: shuffle_batch[x])
    pred_text = []
    for i in range(len(idx)):
        if evals:
            text = []
            for j in range(len(preds[idx[i]])):
                w = preds[idx[i]][j]
                text += [w]
            pred_text += ['%s\t%s\t%s' %(shuffle_batch[idx[i]]+1, evals[idx[i]],
                            '\t'.join(text))]
        else:
            pred_text += ['%s\t%s' %(shuffle_batch[idx[i]]+1, preds[idx[i]])]
    return pred_text

def save_pred(file, preds):
    with open(file, 'w') as f:
        f.write('\n'.join(preds))
    f.close()

def save_config(config, path):
	with open(path, 'w') as f:
	    for key, value in config.__dict__.items():
	        f.write(u'{0}: {1}\n'.format(key, value))
