import os
import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dgllife.model import GCNPredictor
from dgllife.utils import ScaffoldSplitter, RandomSplitter
from sklearn.metrics import roc_curve, auc
from torch import Tensor


class SPLLoss(nn.NLLLoss):
    def __init__(self, *args, n_samples=0, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.threshold = 0.1
        self.growing_factor = 1.3
        self.v = torch.zeros(n_samples).int()

    def forward(self, input: Tensor, target: Tensor, index: Tensor) -> Tensor:
        super_loss = nn.functional.nll_loss(input, target, reduction="none")
        v = self.spl_loss(super_loss)
        self.v[index] = v
        return (super_loss * v).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()


def set_seed(args):
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


def set_model_config(args):
    config = {}
    if args['model'] == 'GCN':
        config = {
            "batch_size": 128,
            "batchnorm": False,
            "dropout": 0.1,
            "gnn_hidden_feats": 128,
            "lr": 0.002,
            "num_gnn_layers": 2,
            "patiene": 30,
            "predictor_hidden_feats": 64,
            "residual": True,
            "weight_decay": 0.001
        }
    return config


def config_update(args, model_config):
    # if args['learning_rate']:
    #     model_config['lr'] = args['learning_rate']
    # if args['batch_size']:
    #     model_config['batch_size'] = args['batch_size']
    # if args['weight_decay']:
    #     model_config['weight_decay'] = args['weight_decay']
    args.update(model_config)
    return args


def split_datset(args, dataset):
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio * args['ratio'], frac_val=val_ratio,
            frac_test=test_ratio + train_ratio * (1 - args['ratio']),
            random_state=args['seed'])
    else:
        return ValueError(f"method {args['split']} is not found!")
    return train_set, val_set, test_set


def load_model(args):
    if args['model'] == 'GCN':
        model = GCNPredictor(
            in_feats=args['in_node_feats'],
            hidden_feats=[args['gnn_hidden_feats']] * args['num_gnn_layers'],
            activation=[F.relu] * args['num_gnn_layers'],
            residual=[args['residual']] * args['num_gnn_layers'],
            batchnorm=[args['batchnorm']] * args['num_gnn_layers'],
            dropout=[args['dropout']] * args['num_gnn_layers'],
            predictor_hidden_feats=args['predictor_hidden_feats'],
            predictor_dropout=args['dropout'],
            n_tasks=args['n_tasks']
        )
    else:
        return ValueError(f"Model {args['model']} is error!")
    return model


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


def criterion(args):
    """
    Set of task type.
    """
    if args['mode'] == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    elif args['mode'] == 'regression':
        return nn.SmoothL1Loss(reduction='none')


def predict(args, model, bg):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats)
    else:
        pass


def plot_train_method(args, loss_list, val_list):
    plt.figure(figsize=(12, 4))
    if args['metric'] in ['roc_auc_score', 'pr_auc_score', 'r2']:
        val_best = max(val_list)
    else:
        val_best = min(val_list)
    plt.subplot(121)
    plt.plot(loss_list, label=f'Best loss = {min(loss_list):.4f}')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.subplot(122)
    plt.plot(val_list, label=f'Best val_score = {val_best:.4f}')
    plt.plot([val_best for i in val_list], linestyle='--')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.suptitle('Train Loss, Validation Score And Test Score in Training Period in ' + args['dataset'])
    plt.savefig(os.path.join(args['result_path'], 'train_val.png'))
    plt.clf()
    return


def plot_result(args, label, predict, score):
    if args['mode'] == 'classification':
        fpr, tpr, threshold = roc_curve(label, predict)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc='lower right')
    else:
        plt.plot([min(label), max(label)], [min(label), max(label)])
        plt.scatter(predict, label, label='{} {:.4f}'.format(args['metric'], score))
        plt.legend(loc='lower right')
    plt.savefig(os.path.join(args['result_path'], 'result.png'))
    plt.clf()
    return
