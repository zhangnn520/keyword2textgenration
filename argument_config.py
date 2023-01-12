#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import argparse

base_dir = os.path.dirname(__file__)

model_name = "checkpoint-68"


def config_argument():
    parser = argparse.ArgumentParser(description='文本生成配置文件')
    parser.add_argument('--eps', type=tuple, default=(1e-30, 1e-3))
    parser.add_argument('--learn_rate', type=float, default=5e-6)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--evaluate_during_training', type=bool, default=True)
    parser.add_argument('--num_of_epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--split_rate', type=float, default=0.9)
    parser.add_argument('--decay_rate', type=float, default=-0.8)
    parser.add_argument('--network', default=model_name, help='specify network')
    parser.add_argument('--data_set_path', type=str, default=os.path.join(base_dir, 'data/demo_data.csv'))
    parser.add_argument('--print_log_step', type=int, default=99)
    parser.add_argument('--clip_threshold', type=float, default=1.0)
    parser.add_argument('--save_train_data_path', type=str,
                        default=os.path.join(base_dir, 'data', 'train_data.json'))
    parser.add_argument('--logging_path', type=str, default=os.path.join(base_dir, 'logs'))
    parser.add_argument('--eval_or_test_data_path', type=str,
                        default=os.path.join(base_dir, 'data', 'eval_or_test_data.json'))
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(base_dir, 'pretrain_model', model_name))
    parser.add_argument('--predict_model_path', type=str,
                        default=os.path.join(base_dir, 'output', 'checkpoint-68'))
    parser.add_argument('--output', type=str, default=os.path.join(base_dir, "output"))
    parser.add_argument('--predict_file_path', type=str,
                        default=os.path.join(base_dir, 'data', 'test_predict.json'))
    args = parser.parse_args()
    return args
