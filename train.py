import os
import time

import numpy as np
import torch
from dgllife.utils import CanonicalAtomFeaturizer, Meter
from torch import nn
from torch.optim import Adam

from dataset import load_data_from_dgl, load_data
from utils import split_datset, set_model_config, config_update, load_model, criterion, predict


def eval_iteration(args, model, val_loader):
    predict_all = []
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(val_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = predict(args, model, bg)
            if len(prediction) == 1:
                predict_all.append(prediction.cpu().numpy()[0])
            else:
                predict_all.extend(prediction.cpu().numpy().squeeze().tolist())
            eval_meter.update(prediction, labels, masks)

    return np.mean(eval_meter.compute_metric(args['metric'])), predict_all


def train_iteration(args, model, train_loader, val_loader, loss_criterion, optimizer):
    model.train()
    best_model = model
    best_score = 0 if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2'] else 999
    iter_count = 0
    loss_list, val_list = [], []
    time_list = []
    time_list.append(time.time())
    while iter_count < args['t_total']:
        for batch_id, batch_data in enumerate(train_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.to(args['device']), masks.to(args['device'])
            prediction = predict(args, model, bg)
            # Mask non-existing labels
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_score, _ = eval_iteration(args, model, val_loader)

            if iter_count % args['print_every'] == 0:
                print(f'iteration {iter_count}/{args["t_total"]}, loss {loss:.4f}, val_score {val_score:.4f}')
            if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2']:
                if val_score > best_score:
                    best_model = model
                    best_score = val_score
            else:
                if val_score < best_score:
                    best_model = model
                    best_score = val_score
            iter_count += 1
            time_list.append(time.time())
            model.train()
            loss_list.append(loss.cpu().detach().numpy())
            val_list.append(val_score)
            if iter_count == args['t_total']:
                break
    np.save(os.path.join(args['result_path'], 'running rime.npy'), np.array(time_list))
    return best_model, best_score, loss_list, val_list


def train(args):
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:' + args['cuda_id'].split(' ')[0])
    else:
        args['device'] = torch.device('cpu')
    args['result_path'] = os.path.join(os.getcwd(), args['result_path'])
    try:
        os.mkdir(args['result_path'])
    except:
        pass
    if args['featurizer_type'] == 'canonical':
        args['node_featurizer'] = CanonicalAtomFeaturizer()
        # args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        args['edge_featurizer'] = None
    if args['featurizer_type'] != 'pre_train':
        args['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            args['in_edge_feats'] = args['edge_featurizer'].feat_size()
    model_config = set_model_config(args)
    args = config_update(args, model_config)
    dataset = load_data_from_dgl(args)
    train_set, val_set, test_set = split_datset(args, dataset)
    args['total'] = int(args['num_epochs'] * len(train_set) / args['batch_size'])
    print('Total Iterations: ', args['t_total'])
    train_loader, val_loader, test_loader = load_data(args, train_set,
                                                      val_set, test_set)
    cuda_id = [int(i) for i in args['cuda_id'].split(' ')]
    if (len(cuda_id) > 1) and (torch.cuda.device_count() > 0):
        model = nn.DataParallel(load_model(args),
                                device_ids=cuda_id)
    else:
        model = load_model(args)
    model.to(args['device'])
    print('Task Type: ', args['mode'])
    loss_criterion = criterion(args)
    print('Loss Function: ', loss_criterion)
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_model, best_score, loss_list, val_list = train_iteration(args, model, train_loader, val_loader, loss_criterion,
                                                                  optimizer)
    plot_train_method(args, loss_list, val_list)
