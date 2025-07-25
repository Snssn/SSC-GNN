import random
import torch.nn as nn
import torch.nn.functional as F
from model.neigh_gen_layers import GraphConvolution
import torch
import numpy as np
import scipy.sparse as sp
from torch.fft import fft, ifft
from scipy.signal import butter, filtfilt


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

class Gen(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(Gen, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        # Discriminator is used to judge the similarity between the generated guidance node features and the positive and negative samples
        self.discriminator = Discriminator(2 * feat_shape, 1)
        
        self.fc1 = nn.Linear(latent_dim, 512).requires_grad_(True)
        self.fc2 = nn.Linear(512, 2048).requires_grad_(True)
        self.fc_flat = nn.Linear(2048, latent_dim).requires_grad_(True)
        self.bn0 = nn.BatchNorm1d(latent_dim).requires_grad_(False)
        self.dropout = dropout

    def forward(self, x):
        x = self.bn0(x)
        raw_feats = torch.tanh(x)
        x = (self.fc1(x))
        #x = self.bn1(x)
        x = (self.fc2(x))
        #x = self.bn2(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x, raw_feats

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        # context-level
        self.f_k = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        # local-level
        self.f_k_env = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def get_contrast_sample(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        classify_label_1 = []
        classify_label_0 = []
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1, device=labels.device):
                classify_label_1.append(i)
            else:
                classify_label_0.append(i)

        gen_feats_spectral = fft(gen_feats, dim=-1).real.float().to(gen_feats.device)
        raw_feats_spectral = fft(raw_feats, dim=-1).real.float().to(raw_feats.device)

        if gen_feats_spectral.shape[-1] != gen_feats.shape[-1]:
            gen_feats_spectral = gen_feats_spectral.mean(dim=-1, keepdim=True)
            raw_feats_spectral = raw_feats_spectral.mean(dim=-1, keepdim=True)

        positive_feats = torch.mean(gen_feats_spectral[classify_label_1], dim=0, keepdim=True)
        negative_feats = torch.mean(gen_feats_spectral[classify_label_0], dim=0, keepdim=True)
        
        raw_positive_feats = torch.mean(raw_feats_spectral[classify_label_1], dim=0, keepdim=True)
        raw_negative_feats = torch.mean(raw_feats_spectral[classify_label_0], dim=0, keepdim=True)

        laplacian = torch.eye(gen_feats_spectral.shape[-1], device=gen_feats.device) - torch.diag_embed(gen_feats_spectral.mean(dim=0))  
        high_pass_feats = torch.matmul(laplacian, gen_feats_spectral.transpose(0, 1)).transpose(0, 1)
        low_pass_feats = gen_feats_spectral - high_pass_feats

        positive_sample = []
        negative_sample = []
        raw_positive_sample = []
        raw_negative_sample = []
        weight = 0.001
        
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1):
                high_freq_feats = high_pass_feats[classify_label_1].mean(dim=0, keepdim=True)
                high_freq_feats = ifft(high_freq_feats, dim=-1).real.float()
                positive_sample.append(weight * high_freq_feats + (1 - weight) * positive_feats)
                
                high_freq_raw_feats = high_pass_feats[classify_label_1].mean(dim=0, keepdim=True)
                high_freq_raw_feats = ifft(high_freq_raw_feats, dim=-1).real.float()
                raw_positive_sample.append(weight * high_freq_raw_feats + (1 - weight) * raw_positive_feats)
                
                low_freq_feats = low_pass_feats[classify_label_0].mean(dim=0, keepdim=True)
                low_freq_feats = ifft(low_freq_feats, dim=-1).real.float()
                negative_sample.append(weight * low_freq_feats + (1 - weight) * negative_feats)
                
                low_freq_raw_feats = low_pass_feats[classify_label_0].mean(dim=0, keepdim=True)
                low_freq_raw_feats = ifft(low_freq_raw_feats, dim=-1).real.float()
                raw_negative_sample.append(weight * low_freq_raw_feats + (1 - weight) * raw_negative_feats)
            else:
                high_freq_feats = high_pass_feats[classify_label_0].mean(dim=0, keepdim=True)
                high_freq_feats = ifft(high_freq_feats, dim=-1).real.float()
                positive_sample.append(weight * high_freq_feats + (1 - weight) * negative_feats)
                
                high_freq_raw_feats = high_pass_feats[classify_label_0].mean(dim=0, keepdim=True)
                high_freq_raw_feats = ifft(high_freq_raw_feats, dim=-1).real.float()
                raw_positive_sample.append(weight * high_freq_raw_feats + (1 - weight) * raw_negative_feats)
                
                low_freq_feats = low_pass_feats[classify_label_1].mean(dim=0, keepdim=True)
                low_freq_feats = ifft(low_freq_feats, dim=-1).real.float()
                negative_sample.append(weight * low_freq_feats + (1 - weight) * positive_feats)
                
                low_freq_raw_feats = low_pass_feats[classify_label_1].mean(dim=0, keepdim=True)
                low_freq_raw_feats = ifft(low_freq_raw_feats, dim=-1).real.float()
                raw_negative_sample.append(weight * low_freq_raw_feats + (1 - weight) * raw_positive_feats)
                
        positive_sample = torch.cat(positive_sample).to(gen_feats.device)
        negative_sample = torch.cat(negative_sample).to(gen_feats.device)
        raw_positive_sample = torch.cat(raw_positive_sample).to(gen_feats.device)
        raw_negative_sample = torch.cat(raw_negative_sample).to(gen_feats.device)

        # local-level
        env_contrast_sample = (2 * env_feats + env_raw_feats) / 3

        return positive_sample, negative_sample, raw_positive_sample, raw_negative_sample, env_contrast_sample

    def get_contrast_loss(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        """
        :param gen_feats: [batchsize, 64]
        :param labels: labels
        :return: loss
        """
        gen_loss = []
        positive_sample, negative_sample, raw_positive_sample, raw_negative_sample, env_contrast_sample \
            = self.get_contrast_sample(gen_feats, raw_feats, env_feats, env_raw_feats, labels)

        # Ensure dimensions of input tensors match with f_k
        positive_sample = positive_sample.view(-1, self.f_k.in1_features)
        gen_feats = gen_feats.view(-1, self.f_k.in1_features)
        raw_positive_sample = raw_positive_sample.view(-1, self.f_k.in1_features)
        negative_sample = negative_sample.view(-1, self.f_k.in1_features)
        raw_negative_sample = raw_negative_sample.view(-1, self.f_k.in1_features)

        gen_loss.append(2 * self.f_k(positive_sample, gen_feats) + self.f_k(raw_positive_sample, gen_feats))
        gen_loss.append(2 * self.f_k(negative_sample, gen_feats) + self.f_k(raw_negative_sample, gen_feats))
        context_logits = torch.cat(tuple(gen_loss))

        env_logits = self.f_k_env(env_contrast_sample, gen_feats)

        return context_logits, env_logits

    def forward(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        """
        contrastive learning loss
        """
        return self.get_contrast_loss(gen_feats, raw_feats, env_feats, env_raw_feats, labels)
