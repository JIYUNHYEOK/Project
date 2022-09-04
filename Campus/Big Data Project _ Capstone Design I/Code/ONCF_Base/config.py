import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--mode', type=str, default="train",
                                  help='train / test')
    parser.add_argument('--model-path', type=str, default="./model_")
    parser.add_argument('--data-path', type=str, default="./data/ml/")
    parser.add_argument('--data-shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--val-step', type=int, default=5)
    parser.add_argument('--test-epoch', type=int, default=10)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--neg-cnt', type=int, default=100)
    parser.add_argument('--at-k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--model', type=str, default='ONCF')

    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--layers', default=[32,32,16,8])
    parser.add_argument('--outer-layers', default=[1,16,16,16,16])
    parser.add_argument('--conv-layers', default=[2,32,16,8])
    #parser.add_argument('--conv-layers', default=[2,16,16,16,16])
    parser.add_argument('--user-cnt', type=int, default=662)
    parser.add_argument('--item-cnt', type=int, default=3883)

    parser.add_argument('--train-path', type=str, default='train_score.pkl')
    parser.add_argument('--val-path', type=str, default='val_score.pkl')
    parser.add_argument('--test-path', type=str, default='test_score.pkl')
    parser.add_argument('--neg-path', type=str, default='neg_score.npy')

    args = parser.parse_args()

    return args
