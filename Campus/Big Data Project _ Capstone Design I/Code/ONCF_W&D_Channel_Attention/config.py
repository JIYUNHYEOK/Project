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
    parser.add_argument('--at-k', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--model', type=str, default='ONCF')

    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--layers', default=[32,32,16,8])
    parser.add_argument('--outer-layers', default=[4,16,16,16,16])
    parser.add_argument('--conv-layers', default=[2,32,16,8])
    #parser.add_argument('--conv-layers', default=[2,16,16,16,16])
    parser.add_argument('--user-cnt', type=int, default=200)
    parser.add_argument('--item-cnt', type=int, default=2928)

    parser.add_argument('--train-path', type=str, default='train_score.pkl')
    parser.add_argument('--val-path', type=str, default='val_score.pkl')
    parser.add_argument('--test-path', type=str, default='test_score.pkl')
    parser.add_argument('--neg-path', type=str, default='neg_score.npy')

    # side_info
    ## user
    parser.add_argument('--train_user-path', type=str, default='train_user_score.pkl')
    parser.add_argument('--val_user-path', type=str, default='val_user_score.pkl')
    parser.add_argument('--test_user-path', type=str, default='test_user_score.pkl')
    parser.add_argument('--neg_user-path', type=str, default='neg_user_score.npy')
    ## movie
    parser.add_argument('--train_movie-path', type=str, default='train_movie_score.pkl')
    parser.add_argument('--val_movie-path', type=str, default='val_movie_score.pkl')
    parser.add_argument('--test_movie-path', type=str, default='test_movie_score.pkl')
    parser.add_argument('--neg_movie-path', type=str, default='neg_movie_score.npy')

    parser.add_argument('--lst_user', type=list, default=[0,1])
    parser.add_argument('--lst_movie', type=list, default=[0,1])
    parser.add_argument('--lst_un', type=list, default=[0,1])

    parser.add_argument('--deep_column_user_idx', type=dict, default={'Occupation': 1, 'Zip-code': 2, 'Age': 3})
    parser.add_argument('--deep_column_movie_idx', type=dict, default={'Title': 1, 'Genres': 2})
    parser.add_argument('--user_embeddings_input', type=list, default=[('Zip-code', 530, 12), ('Occupation', 21, 8)])
    parser.add_argument('--movie_embeddings_input', type=list, default=[('Title', 3883, 16), ('Genres', 301, 16)])
    parser.add_argument('--user_continuous_cols', type=list, default=["Age"])
    parser.add_argument('--movie_continuous_cols', type=list, default=[])
    parser.add_argument('--hidden_layers', type=list, default=[100,50])
    parser.add_argument('--user_wide_dim', type=int, default=253)
    parser.add_argument('--movie_wide_dim', type=int, default=3224)


    args = parser.parse_args()

    return args
