# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:08:59 2022

@author: LSH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Three different activation function uses in the ZINB-based denoising autoencoder.
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e4)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e3)
PiAct = lambda x: 1/(1+torch.exp(-1 * x))

# A general GCN layer.
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = torch.tanh(output)
        return output
    
# A dot product operation uses in the decoder of GAE. 
def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

# A random Gaussian noise uses in the ZINB-based denoising autoencoder.
class GaussianNoise(nn.Module):
    def __init__(self, device, sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device = device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

# A multi-head attention layer has two different input (query and key/value).
class AttentionWide(nn.Module):
    def __init__(self, emb, p = 0.2, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask
        self.dropout = nn.Dropout(p)
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, y):
        b = 1
        t, e = x.size()
        h = self.heads
        # assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.dropout(self.toqueries(y)).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        self.attention_weights = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

# Final model
class AttentionAE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, device):
        super(AttentionAE, self).__init__()

        # autoencoder for intra information
        #self.dropout = nn.Dropout(0.2)
        self.Gnoise = GaussianNoise(sigma=0.1, device=device)
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN_1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN_2 = nn.BatchNorm1d(n_enc_2)
        self.z_layer = nn.Linear(n_enc_2, n_z)
        
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
   
        self.calcu_pi = nn.Linear(n_dec_2, n_input)
        self.calcu_disp = nn.Linear(n_dec_2, n_input)
        self.calcu_mean = nn.Linear(n_dec_2, n_input)
       
        self.gnn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_2 = GNNLayer(n_enc_2, n_z)

        self.attn1 = AttentionWide(n_enc_2)
        self.attn2 = AttentionWide(n_z)


    def forward(self, x, adj):    
        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        # enc_h1 = (self.attn1(enc_h1, h1)).squeeze(0) + enc_h1
        h1 = self.gnn_1(enc_h1, adj)
        h2 = self.gnn_2(h1, adj)
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        enc_h2 = (self.attn1(enc_h2, h1)).squeeze(0)+enc_h2
        z = self.z_layer(self.Gnoise(enc_h2))
        z = (self.attn2(z, h2)).squeeze(0)+z
        #decoder
        A_pred = dot_product_decode(h2)
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        
        pi = PiAct(self.calcu_pi(dec_h2))
        mean = MeanAct(self.calcu_mean(dec_h2))
        disp = DispAct(self.calcu_disp(dec_h2))        
        return z, A_pred, pi, mean, disp