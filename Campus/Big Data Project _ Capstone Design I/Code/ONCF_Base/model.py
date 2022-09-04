import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class GMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(GMF, self).__init__()
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        return torch.mul(u, v)

class ONCF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(ONCF, self).__init__()

        self.gmf = GMF(num_user, num_item, emb_dim)
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        # convs = []
        # for (in_d, out_d) in zip(layers[:-1], layers[1:]):
        #     convs.append(nn.Conv2d(in_d, out_d, 2, 2))
        #     convs.append(nn.ReLU())
        # self.conv = nn.Sequential(*convs)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 2, 2),
            nn.ReLU(inplace=True),
        )
        self.predict= nn.Linear(emb_dim+layers[-1], 1)

    def forward(self, u, v, n):
        # GMF
        gmf = self.gmf(u,v)
        gmf_n=self.gmf(u.unsqueeze(1).expand_as(n),n).view(-1,gmf.size(-1))

        # MLP
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n)
        ## outer product
        x = torch.bmm(u.unsqueeze(2), v.unsqueeze(1))
        x_n=torch.bmm(u.repeat(1,n.size(1)).view(-1,n.size(-1),1),
                      n.view(-1,1,n.size(-1)))

        h = self.conv(x.unsqueeze(1)).squeeze()
        h_n=self.conv(x_n.unsqueeze(1)).squeeze()

        # Fusion
        pred = self.predict(torch.cat((gmf,h), 1)).view(-1)
        pred_n=self.predict(torch.cat((gmf_n,h_n), 1)).view(-1)

        return pred, pred_n