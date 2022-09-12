'''
Functions for k-fold evaluation of models.
'''

import random
import pickle
import numpy as np
from sklearn.model_selection import KFold
import torch

import proposed_method.data_utils as data_utils
from proposed_method.MetaRegGNN import MetaRegGNN
from collections import OrderedDict
from config import Config

def evaluate_MetaRegGNN(shuffle=False, random_state=None,
                    dropout=0.1, k_list=list(range(2, 16)), lr=1e-3, wd=5e-4,
                    device=torch.device('cpu'), num_epoch=100):


    overall_preds = {k: [] for k in k_list}
    overall_scores = {k: [] for k in k_list}
    train_mae = {k: [] for k in k_list}

    data = data_utils.load_dataset_pytorch()
    fold = -1
    for train_idx, test_idx in KFold(Config.K_FOLDS, shuffle=shuffle,
                                     random_state=random_state).split(data):
        fold += 1
        print(f"Cross Validation Fold {fold+1}/{Config.K_FOLDS}")
        for k in k_list:
            selected_train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            candidate_model = MetaRegGNN(116, 64, 1, dropout).float().to(device)
            optimizer = torch.optim.Adam(candidate_model.parameters(), lr=Config.MetaRegGNN.ETA, weight_decay=wd)
            train_loader, test_loader = data_utils.get_loaders(selected_train_data, test_data)
            candidate_model.train()
            for epoch in range(num_epoch):
                tgt_data = iter(test_loader)
                preds = []
                scores = []
                outer_loss = torch.tensor(0., device=device)
                for i,(batch_src) in enumerate(train_loader):                    
                    out_src = candidate_model(batch_src.x.to(device), data_utils.to_dense(batch_src).adj.to(device))
                    inner_loss = candidate_model.loss(out_src.view(-1, 1), batch_src.y.to(device).view(-1, 1))
                    candidate_model.zero_grad()
                    params = OrderedDict(candidate_model.named_parameters())
                    grads = torch.autograd.grad(inner_loss,
                                params.values(),
                                create_graph=True)
                    updated_params = OrderedDict()
                    for (name, param), grad in zip(params.items(), grads):
                        updated_params[name] = param - Config.MetaRegGNN.GAMMA * grad
                    candidate_model.load_state_dict(updated_params)
                    try:
                        batch_tgt = next(tgt_data)
                    except StopIteration:
                        tgt_data = iter(test_loader)
                        batch_tgt = next(tgt_data)
                    out_tgt = candidate_model(batch_tgt.x.to(device), data_utils.to_dense(batch_tgt).adj.to(device),params)
                    outer_loss = candidate_model.loss(out_tgt.view(-1, 1), batch_tgt.y.to(device).view(-1, 1))
                    if i%5==0:  #number of shots                 
                        outer_loss.backward()
                        optimizer.step()

                    preds.append(out_src.cpu().data.numpy())
                    scores.append(batch_src.y.long().numpy())
                    
                    
                preds = np.hstack(preds)
                scores = np.hstack(scores)
                epoch_mae = np.mean(np.abs(preds.reshape(-1, 1) - scores.reshape(-1, 1)))
                train_mae[k].append(epoch_mae)
            
            candidate_model.eval()
            with torch.no_grad():
                preds = []
                scores = []
                for batch in test_loader:
                    out = candidate_model(batch.x.to(device), data_utils.to_dense(batch).adj.to(device))

                    loss = candidate_model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.cpu().long().numpy())

                preds = np.hstack(preds)
                scores = np.hstack(scores)

            overall_preds[k].extend(preds)
            overall_scores[k].extend(scores)

    for k in k_list:
        overall_preds[k] = np.vstack(overall_preds[k]).ravel()
        overall_scores[k] = np.vstack(overall_scores[k]).ravel()


    overall_preds = overall_preds[k_list[0]]
    overall_scores = overall_scores[k_list[0]]

    return overall_preds, overall_scores, train_mae