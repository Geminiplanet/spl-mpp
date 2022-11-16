import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model import GCNPredictor
from dgllife.utils import ScaffoldSplitter, RandomSplitter


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
