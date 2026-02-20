import torch
from torch import nn, optim
import torch.nn.functional as F
import math 


class InterpositionAttention(nn.Module):
    def __init__(self, args, in_c, out_c, num_id, grap_size, IA_emb, dropout):
        """
        Inductive Attention
        :param in_c: embeding size
        :param out_c: embeding output size
        :param num_id: number of nodes
        :param grap_size: node embeding size
        :param IA_emb: node embedding in IA
        :param dropout: dropout
        """
        super(InterpositionAttention, self).__init__()
        self.in_c = in_c  # number of input feature
        self.out_c = out_c  # number of output feature
        self.num_id = num_id
        self.drop = dropout
        self.args = args
        self.dropout = nn.Dropout(dropout)
        self.active_mode = self.args.active_mode  # sprtrelu / adptpolu
        self.act_k = self.args.act_k
        if self.active_mode == "adptpolu":
            self.poly_coefficients = nn.Parameter(torch.randn(self.num_id, 1, self.act_k+1), requires_grad=True)
            # self.poly_coefficients = nn.Parameter(torch.randn(self.num_id, 1, self.act_k+1) * 1e-2, requires_grad=True)
        self.dk = 64

        self.W = nn.Parameter(torch.FloatTensor(size=(num_id, in_c, out_c)))
        for w in self.W:
            nn.init.xavier_uniform_(w)  # initialize
        self.a = nn.Parameter(torch.FloatTensor(size=(num_id, out_c, 1)))
        for a in self.a:
            nn.init.xavier_uniform_(a)  # initialize

        # leakyrelu
        self.leakyrelu = nn.LeakyReLU()  # when x<0,alpha*x
        # self.GL = nn.Parameter(torch.FloatTensor(num_id, grap_size))
        self.GL = IA_emb
        nn.init.kaiming_uniform_(self.GL)

    def forward(self, inp):
        """
        inp: input_fea [B, N, C]
        """
        ### federated attention 
        hw = torch.einsum('bnc,nco->bno', inp, self.W) 
        att_raw = torch.einsum('bno,noe->bne', hw, self.a) / math.sqrt(self.dk)
        b = torch.exp(att_raw)  # [B,N, 1]
        
        bhw = b * hw
        if self.active_mode == 'sprtrelu':
            transformed_E = torch.relu(self.GL)  #sprtrelu
            transformed_E_T = transformed_E.transpose(0, 1)
            EB = torch.einsum('dn,bne->bde', transformed_E_T, b)
            EBHW = torch.einsum('dn,bno->bdo', transformed_E_T, bhw)
            numerator = bhw + torch.einsum('nd,bdo->bno', transformed_E, EBHW)
            denominator = b + torch.einsum('nd,bde->bne', transformed_E, EB)
        else:
            pos_GL = torch.relu(self.GL)
            coeffs = F.softplus(self.poly_coefficients)
            transformed_E = [self.transform(k, pos_GL) for k in range(self.act_k + 1)]
            EB = [torch.einsum('dn,bne->bde', e.transpose(0, 1), b) for e in transformed_E]
            EBHW = [torch.einsum('dn,bno->bdo', e.transpose(0, 1), bhw) for e in transformed_E]
            EEB = torch.stack([torch.einsum("nd,bde->bne", transformed_E[i], EB[i]) for i in range(self.act_k + 1)])
            EEBHW = torch.stack([torch.einsum("nd,bdo->bno", transformed_E[i], EBHW[i]) for i in range(self.act_k + 1)])
            numerator = bhw + torch.concat([torch.einsum('ak,kbno->abno', coeffs[cid], EEBHW[:, :, cid:cid+1, :])[0] for cid in range(self.num_id)], dim = 1)
            denominator = b + torch.concat([torch.einsum('ak,kbne->abne', coeffs[cid], EEB[:, :, cid:cid+1, :])[0] for cid in range(self.num_id)], dim = 1)
            
        denominator = denominator.clamp(min=1e-6)
        h_prime = F.relu(numerator / denominator)
        h_prime = torch.nan_to_num(h_prime, nan=0.0, posinf=1e6, neginf=-1e6)

        return h_prime
        
    def cartesian_prod(self, A, B):
        transformed = torch.stack(list(map(torch.cartesian_prod, A, B)))
        transformed = transformed[...,0] * transformed[...,1]
        return transformed

    def transform(self, k, E):
        ori_k = k
        transformed = torch.ones(E.shape[0], 1).to(E.device)
        cur_pow = self.cartesian_prod(transformed, E)
        while k > 0:
            if k % 2 == 1:
                transformed = self.cartesian_prod(transformed, cur_pow)
            cur_pow = self.cartesian_prod(cur_pow, cur_pow)
            k //= 2
        assert transformed.shape[0] == E.shape[0], (transformed.shape[0], E.shape[0])
        assert transformed.shape[1] == E.shape[1]**ori_k, (transformed.shape[1], E.shape[1], ori_k)
        return transformed
