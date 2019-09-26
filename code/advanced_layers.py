import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time

def kl_divergence(p, q):
    return p * tf.log(tf.maximum(1.e-10, p/q))

torch.manual_seed(123)

class Ranker(nn.Module):
    def __init__(self, args, emb, lr = 0.1, dropout = 0.2):
        super(Ranker, self).__init__()
        self.args = args
        self.emb = emb
        self.vocab_size, self.emb_dim = emb.shape
        n_d = args.hidden_dimension

        self.emb_init = nn.Embedding(self.emb.shape[0], self.emb.shape[1])
        self.emb_init.weight.data.copy_(torch.from_numpy(self.emb))
        self.encode_linear= nn.Sequential(nn.Linear(self.emb_dim, n_d), nn.ReLU())
        self.compare_rnn1 = nn.LSTM(n_d, n_d, dropout = dropout, batch_first=True)
        self.Linear_layers = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(n_d, 1))
        self.compare_layers = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(2*n_d+1, n_d))
        self.Z_linear = nn.Sequential(nn.Linear(n_d, 1), nn.Sigmoid())

        self.lr = lr
        self.dropout = dropout

    def encoder(self, question):
        '''
        Convert input question to question embedding
        question
        '''
        padding_id = self.padding_id = 0
        B, QL = question.size()
        n_d = self.args.hidden_dimension

        self.emb_init.weight.data[0, :].fill_(0)
        emb_question = self.inputs = self.emb_init(question)
        self.question_mask = x_masks = 1 - torch.eq(question, padding_id).type(torch.FloatTensor)
        emb_question = self.encode_linear(emb_question) # non-linear transformation
        emb_question = emb_question.view(B, -1, QL, n_d)
        return emb_question

    def ranker(self, emb_question, relation, dep_hc):
        '''
        Rank stories for each iteration
        Input:
        1) emb_question: the questin with embeddings (batch_size, top ranked question, question length, embedding dimension)
        2) relation: the relation at this iteration (batch_size, relation number, relation length)
        3) dep_hc: $a^{(t-1)}_i$ (batch_size, relation number, question length, 1)
        Output:
        1) preds: matching probability of relations at this iteration
        2) emb_question: the questin with embeddings
        3) stop: stop signal $\bar{z}^{(t)}$
        4) dep_hc: $a^{(t)}_i$
        '''
        padding_id = self.padding_id = 0
        n_d = self.args.hidden_dimension
        n_e = self.emb_dim
        B, SN, SL = relation.size() # (batch_size, relation number, relation length)
        _, top, QL, _ = emb_question.size() # (batch_size, top ranked question, question length, embedding dimension)

        emb_relation = self.emb_init(relation.contiguous().view(-1, SL)).view(-1, SN, SL, n_e)
        emb_relation = emb_relation.view(-1, SL, n_e)
        emb_relation = self.encode_linear(emb_relation) # non-linear transformation
        emb_relation = emb_relation.view(B,1,SN,SL,n_d).repeat(1,top,1,1,1).view(-1,SL,n_d)
        emb_question = emb_question.view(-1, QL, n_d)

        trans_emb_question = torch.transpose(emb_question, 1, 2)
        trans_emb_relation = emb_relation.view(-1, SN*SL, n_d)
        matrix_e = torch.matmul(trans_emb_relation, trans_emb_question).view(-1, SL, QL) # obtain attention of question and story (Eq (1))

        relation_mask = 1 - torch.eq(relation, padding_id).type(torch.FloatTensor).view(-1, SN*SL)
        relation_mask = torch.unsqueeze(relation_mask, 2) # relation mask
        question_mask = torch.unsqueeze(self.question_mask, 1) # question mask
        mask_matrix_e = torch.matmul(relation_mask, question_mask).view(B, SN, SL, QL)
        mask_matrix_e = torch.unsqueeze(mask_matrix_e,1).repeat(1,top,1,1,1).view(-1,SL,QL).cuda()

        mask_matrix_e_values = -1e10*torch.ones_like(matrix_e).cuda()
        matrix_e = mask_matrix_e*matrix_e + (1-mask_matrix_e)*mask_matrix_e_values # attention with mask
        matrix_alpha = F.softmax(matrix_e, 1) # obtain normalized attention along relations (Eq (2))
        matrix_beta = F.softmax(matrix_e, 2) # obtain normalized attention along questions (Eq (3))
        mask_matrix_e_values = 0.*torch.ones_like(matrix_e)
        matrix_alpha = mask_matrix_e*matrix_alpha + (1-mask_matrix_e)*mask_matrix_e_values
        matrix_alpha = torch.transpose(matrix_alpha, 1, 2) # (-1, question length, relation length)
        matrix_beta = mask_matrix_e*matrix_beta + (1-mask_matrix_e)*mask_matrix_e_values # (-1, relation length, question length)

        soft_emb_question = torch.matmul(matrix_alpha, emb_relation).view(-1, SN, QL, n_d) # obtain aligned questions using sotries
        trans_emb_question = emb_question.view(-1, 1, QL, n_d).repeat(1, SN, 1, 1)

        matrix_beta = torch.sum(matrix_beta, 1).view(-1, SN, QL, 1) # Accumulate attention for question at this iteration
        dep_hc = dep_hc.view(-1, 1, QL, 1).repeat(1, SN, 1, 1).view(-1, SN, QL, 1)
        dep_hc = dep_hc + matrix_beta  # Accumulate attention for question (a_i^(t)) with previous iterations
        matrix_m = torch.cat((trans_emb_question * soft_emb_question,
                            (trans_emb_question - soft_emb_question)**2,
                             dep_hc), 3) # arrange matching vectors (Eq (4))
        matrix_m = matrix_m.view(-1, QL, 2*n_d+1)
        matrix_m = self.compare_layers(matrix_m)
        matrix_m, _ = self.compare_rnn1(matrix_m) # derive single vector $\bar{m}^{(t)}$ (Eq (5))
        bar_matrix_m = torch.max(matrix_m, 1)[0]
        dep_hc = dep_hc.view(B, -1, QL, 1)

        gamma = self.Linear_layers(bar_matrix_m).view(B, -1, SN) # drive $\gamma^{(t)}$ (Eq (6))
        preds = F.softmax(gamma, -1) # normalize $\gamma^{(t)}$ along all stories
        probs = torch.squeeze(self.Z_linear(bar_matrix_m), -1).view(-1, SN) # derive $z^{(t)}(p)$
        emb_question = emb_question.view(-1, 1, QL, n_d).repeat(1, SN, 1, 1)
        emb_question = emb_question.view(B, -1, QL, n_d)
        stop = 1. - torch.min(probs) # derive $bar{z}^{(t)}$
        return preds, emb_question, stop, dep_hc

    def final_probs(self, preds, prev_preds, s):
        '''
        Generate final probability
        Input:
        1) preds: $\gamma^{(t)}$
        2) prev_preds: $s^{(t-1)}$
        3) relation:
        '''
        B = preds.size(0)
        s = torch.sum(s, 2)
        mask_alig = 1 - torch.eq(s, self.padding_id).type(torch.FloatTensor).cuda()
        mask_alig_values = 0*torch.ones_like(mask_alig).cuda()

        prev_preds = torch.unsqueeze(prev_preds, 2)
        preds = (prev_preds*preds).view(B, -1) # Generate final proability (Eq (7))

        preds = mask_alig*preds + (1-mask_alig)*mask_alig_values # Mask the empty relations
        return preds

    def obtain_reward(self, y, preds, step_loss):
        '''
        Generate final loss
        Input:
        1) y: ground truth
        2) preds: predicted proability
        3) step_loss: loss caused by the wrong step prediction if any
        '''
        #loss_mat = self.loss_mat = F.kl_div(preds, y, size_average=False, reduce=True)
        loss_mat = self.loss_mat = torch.sum((y - preds)**2)
        print('step loss %s\t loss mat %s' %(step_loss.cpu().data.numpy(), loss_mat.cpu().data.numpy()))
        self.loss = loss = loss_mat + step_loss
        return loss
