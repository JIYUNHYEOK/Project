# Importing the libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable

from model import *
from config import get_args
from data_loader import get_loader

from collections import namedtuple
import pickle

args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# UserID::MovieID::Rating::Timestamp (5-star scale)
train_loader = get_loader(args.data_path, args.train_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle)
val_loader = get_loader(args.data_path, args.val_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle)
test_loader = get_loader(args.data_path, args.test_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle)

# Getting the number of users and movies
if 'ml' in args.data_path:
    num_users  = 662
    num_movies = 3883

# Creating the architecture of the Neural Network
if args.model == 'GMF':
    model = GMF(num_users, num_movies, args.emb_dim)
elif args.model == 'ONCF':
    model = ONCF(num_users, num_movies, args.emb_dim, args.outer_layers)

if torch.cuda.is_available():
    model.cuda()
"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
# print(model)
# print("The number of parameters: {}".format(num_params))

criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

best_epoch = 0
best_loss  = 9999.

tr_loss = []
v_loss = []
v_hit = []

def train():
    global best_loss, best_epoch, tr_loss, v_loss, v_hit
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                              'model-%d.pkl'%(args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = 0
        model.train()
        for s_train, (x, n) in enumerate(train_loader):
            x = x.long().to(device)
            n = n.long().to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            #r = Variable(x[:,2]).float()

            pred, neg_pred = model(u, v, n)
            # loss = criterion(pred, torch.ones_like(pred).to(device)) \
            #      + criterion(neg_pred, torch.zeros_like(neg_pred).to(device))
            # train_loss += loss.item()
            loss_w = criterion(pred, torch.ones_like(pred).to(device))
            loss_n = criterion(neg_pred, torch.zeros_like(neg_pred).to(device))
            
            train_loss += (loss_w.item() + loss_n.item())
            
            # print(f'watch: {criterion(pred, torch.ones_like(pred))}')
            # print(f'unwatch: {criterion(neg_pred, torch.zeros_like(neg_pred))}')

            model.zero_grad()
            # loss.backward()
            loss_n.backward(retain_graph = True)
            loss_w.backward()
            optimizer.step()
            
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/(s_train+1)))
        tr_loss.append(train_loss/(s_train+1))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0
            val_hits = 0
            with torch.no_grad():
                for s_valid, (x, n) in enumerate(val_loader):
                    x = x.long().to(device)
                    n = n.long().to(device)
                    u = Variable(x[:,0])
                    v = Variable(x[:,1])
                    #r = Variable(x[:,2]).float()

                    pred, neg_pred = model(u, v, n)
                    # loss = criterion(pred, torch.ones_like(pred).to(device)) \
                    #      + criterion(neg_pred, torch.zeros_like(neg_pred).to(device))
                    # val_loss += loss.item()
                    
                    loss_w = criterion(pred, torch.ones_like(pred).to(device))
                    loss_n = criterion(neg_pred, torch.zeros_like(neg_pred).to(device))
                    
                    val_loss += (loss_w.item() + loss_n.item())
                    
                    print(f'watch: {criterion(pred, torch.ones_like(pred))}')
                    print(f'unwatch: {criterion(neg_pred, torch.zeros_like(neg_pred))}')

                    # Hit Ratio
                    pred = torch.cat((pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
                    _, topk = torch.sort(pred, 1, descending=True)
                    val_hits += sum([0 in topk[k, :args.at_k] for k in range(topk.size(0))])

            print('[val loss] : '+str(val_loss/(s_valid+1))+' [val hit ratio] : '+str(val_hits/num_users))
            v_loss.append(val_loss/(s_valid+1))
            v_hit.append(val_hits/num_users)


            if best_loss > (val_loss/(s_valid+1)):
                best_loss = (val_loss/(s_valid+1))
                best_epoch= epoch+1
                torch.save(model,
                       os.path.join(args.model_path+args.model,
                       'model-%d.pkl'%(epoch+1)))

te_val = []
te_hit = []

def test():
    global te_val, te_hit
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                          'model-%d.pkl'%(best_epoch))).state_dict())
    model.eval()
    test_loss = 0
    test_hits = 0
    with torch.no_grad():
        for s, (x, n) in enumerate(test_loader):
            x = x.long().to(device)
            n = n.long().to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            #r = Variable(x[:,2]).float()

            pred, neg_pred = model(u, v, n)
            loss = criterion(pred, torch.ones_like(pred).to(device)) \
                 + criterion(neg_pred, torch.zeros_like(neg_pred).to(device))
            test_loss += loss.item()

            # Hit Ratio
            pred = torch.cat((pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
            _, topk = torch.sort(pred, 1, descending=True)
            test_hits += sum([0 in topk[k, :args.at_k] for k in range(topk.size(0))])

    print('[test loss] : '+str(test_loss/(s+1))+' [test hit ratio] : '+str(test_hits/num_users))
    te_val.append(test_loss/(s+1))
    te_hit.append(test_hits/num_users)

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()

    all_loss = pd.DataFrame({'tr_loss' : [tr_loss], 'v_loss':[v_loss], 'v_hit' : [v_hit], 'te_val': [te_val], 'te_hit' : [te_hit]})
    all_loss.to_csv('./visual_data/visual_loss.csv')