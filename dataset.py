from functools import partial

from dgllife.data import Tox21
from dgllife.utils import smiles_to_bigraph
from torch.utils.data import DataLoader

from utils import collate_molgraphs


def load_data_from_dgl(args):
    if args['dataset'] == 'Tox21':
        dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'])
        args['mode'] = 'classification'
    args['n_tasks'] = dataset.n_tasks

    return dataset


def load_data(args, train_set, val_set, test_set):
    print('Training method not use Curriculum Learning')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True,
                              collate_fn=collate_molgraphs)
    if val_set:
        val_loader = DataLoader(val_set,
                                batch_size=int(len(val_set) * 0.2) if int(len(val_set) * 0.2) < 1000 else 1000,
                                num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    else:
        val_loader = None
    test_loader = DataLoader(test_set,
                             batch_size=int(len(test_set) * 0.2) if int(len(test_set) * 0.2) < 1000 else 1000,
                             num_workers=args['num_workers'], collate_fn=collate_molgraphs)
    return train_loader, val_loader, test_loader
