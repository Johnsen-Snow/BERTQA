import argparse
import os
import sys

import torch

sys.path.append('../')
from others.logging import init_logger
from train_squad import train, inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', default='../data')
    parser.add_argument('-model_path', default='../models/model.pt')
    parser.add_argument('-bert_dir', default='../bert/bert_base_uncased_english')
    parser.add_argument('-log_file', default='../logs/squad_train2.0.log')
    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-dataset', default='v2.0.json', type=str, choices=['v1.1.json', 'v2.0.json'])
    parser.add_argument('-version_2_with_negative', default=True, type=bool)
    parser.add_argument('-seed', default=12345, type=int)

    parser.add_argument('-batch_size', default=12, type=int)
    parser.add_argument('-hidden_size', default=768, type=int)
    parser.add_argument('-n_best_size', default=10, type=int)
    parser.add_argument('-max_answer_len', default=30, type=int)
    parser.add_argument('-max_sen_len', default=384, type=int)
    parser.add_argument('-max_query_len', default=64, type=int)
    parser.add_argument('-doc_stride', default=128, type=int)
    parser.add_argument('-max_position_embeddings', default=512, type=int)
    parser.add_argument('-is_sample_shuffle', default=True, type=bool)

    parser.add_argument('-lr', default=3.5e-5, type=float)
    parser.add_argument('-epochs', default=2, type=int)
    parser.add_argument('-model_val_per_epoch', default=1, type=int)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    init_logger(args.log_file)
    device = torch.device('cpu' if args.visible_gpus == '-1' else 'cuda')

    train(args, device)
    inference(args, device)
