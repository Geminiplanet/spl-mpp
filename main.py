from argparse import ArgumentParser

from train import train, train_spl


def main():
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset', choices=['Tox21', 'ESOL'], default='Tox21', help='Dataset to use')
    parser.add_argument('-mo', '--model', choices=['GCN'], default='GCN', help='Model to use')
    parser.add_argument('-f', '--featurizer_type', choices=['canonical'], default='canonical',
                        help='Featurization for atoms (and bonds), this is required for models')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'r2', 'mae', 'rmse'],
                        default='roc_auc_score', help='Metric for evaluation')
    parser.add_argument('-sr', '--split_ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of data sets used for training, verification and testing')
    parser.add_argument('-se', '--seed', type=int, default=0, help='Global random seed')
    parser.add_argument('-rp', '--result_path', type=str, default='results', help='Path ro save training results')
    parser.add_argument('-id', '--cuda_id', type=str, default='0', help='cuda id used')

    parser.add_argument('-ne', '--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of processes for data loading')

    parser.add_argument('-pe', '--print_every', type=int, default=50,
                        help='Print the training progress every X mini-batches')

    args = parser.parse_args().__dict__
    print(args)
    train_spl(args)


if __name__ == '__main__':
    main()
