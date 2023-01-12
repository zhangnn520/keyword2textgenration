#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import re
import json
import copy
import math
import time
import torch
import warnings
from tqdm import tqdm
from bert_score.bert_score import score
from argument_config import config_argument
from tools.tools import generate, read_csv, write_json, read_json, logFrame
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Adafactor

warnings.filterwarnings("ignore")
base_dir = os.path.dirname(__file__)
log = logFrame()

args = config_argument()
logger = log.logg(args)
logger.info(json.dumps(vars(args), ensure_ascii=False, indent=4))

if args.do_train:
    # 读取数据，并进行数据集拆分
    train_data, eval_or_test_data, data_set_number = read_csv(args.data_set_path, args.split_rate)
    # 保存模型训练和验证数据集
    write_json(args.save_train_data_path, train_data)
    write_json(args.eval_or_test_data_path, eval_or_test_data)

    train_num_of_batches = int(len(train_data) / args.train_batch_size)
    eval_num_of_batches = int(len(eval_or_test_data) / args.eval_batch_size)

    # 设置设备是否使用gpu或者cpu
    if torch.cuda.is_available() and not args.no_cuda:
        dev = torch.device("cuda:0")
        logger.info("Running on the GPU")
    else:
        dev = torch.device("cpu")
        logger.info("Running on the CPU")
    # 加载模型配置文件
    tokenizer = MT5Tokenizer.from_pretrained(args.model_path)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_path, return_dict=True)
    # 将模型加载进gpu或cpu设备中
    model.to(dev)


def evaluate(arg, eval_model, eval_data_loader, device):
    eval_model.to(device)
    eval_model.eval()
    total_val_loss = 0
    pre_text_list, org_text_list = list(), list()
    for j in tqdm(range(eval_num_of_batches)):
        input_batch, label_batch = list(), list()
        eval_batch_data = eval_data_loader[j * arg.eval_batch_size:j * arg.eval_batch_size + arg.eval_batch_size]
        for index, eval_row in enumerate(eval_batch_data):
            inputs = 'keywords: ' + eval_row['input_text'] + '</s>'
            input_batch.append(inputs)
            labels = eval_row['target_text'] + '</s>'
            label_batch.append(labels)
            org_text_list.append(eval_row['target_text'])
            # 验证集预测，获得结果
        input_batch = tokenizer.batch_encode_plus(
            input_batch, padding=True, max_length=args.max_length, return_tensors='pt')["input_ids"]
        label_batch = tokenizer.batch_encode_plus(
            label_batch, padding=True, max_length=args.max_length, return_tensors="pt")["input_ids"]
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)
        outputs = eval_model(input_ids=input_batch, labels=label_batch)
        result_outputs = eval_model.generate(input_batch, max_length=arg.max_length, min_length=0)
        pre_text = [re.sub('<pad> |</s>|# | <pad>', "", tokenizer.decode(result_out, errors='ignore'))
                    for result_out in result_outputs]
        pre_text_list += pre_text
        loss = outputs.loss
        if isinstance(eval_model, torch.nn.DataParallel):
            loss = loss.mean()
            total_val_loss += loss.mean().item()

    P, R, F1 = score(pre_text_list, org_text_list, lang="zh", verbose=True)
    eval_loss = total_val_loss / int(eval_num_of_batches)
    perplexity = math.exp(eval_loss)
    perplexity = torch.tensor(perplexity)
    logger.info(f"Eval_loss:{total_val_loss},perplexity:{perplexity},F1:{F1.mean():.3f},精确率:"
                f"{P.mean():.3f}，召回率:{R.mean():.3f}")
    return F1.mean().item(), perplexity


def model_train():
    optimizer = Adafactor(model.parameters(),
                          lr=args.learn_rate,
                          eps=args.eps,
                          clip_threshold=args.clip_threshold,
                          decay_rate=args.decay_rate,
                          beta1=None,
                          weight_decay=args.weight_decay,
                          relative_step=False,
                          scale_parameter=False,
                          warmup_init=False)

    model.train()
    for epoch in range(1, args.num_of_epochs + 1):
        start_time = time.time()
        logger.info('Running epoch: {}'.format(epoch))
        running_loss = 0
        for i in range(train_num_of_batches):
            input_batch = []
            label_batch = []
            train_batch_data = train_data[i * args.train_batch_size:i * args.train_batch_size + args.train_batch_size]
            if train_batch_data:
                try:
                    for num, row in enumerate(train_batch_data):
                        inputs = 'keywords: ' + row['input_text'] + '</s>'
                        labels = row['target_text'] + '</s>'
                        input_batch.append(inputs)
                        label_batch.append(labels)
                    input_batch = tokenizer.batch_encode_plus(
                        input_batch, padding=True, max_length=args.max_length, return_tensors='pt')["input_ids"]
                    label_batch = tokenizer.batch_encode_plus(
                        label_batch, padding=True, max_length=args.max_length, return_tensors="pt")["input_ids"]
                except Exception as e:
                    print(e, i)
            else:
                continue
            input_batch = input_batch.to(dev)
            label_batch = label_batch.to(dev)
            # clear out the gradients of all Variables
            optimizer.zero_grad()
            # Forward propogation
            outputs = model(input_ids=input_batch, labels=label_batch)
            loss = outputs.loss
            loss_num = loss.item()
            running_loss += loss_num
            if i % args.print_log_step == 0:
                logger.info(f"total batch num:{train_num_of_batches},batch_num:{i + 1},loss:{loss_num},"
                            f"time:{time.time() - start_time}\n")
            # calculating the gradients
            loss.backward()
            # updating the params
            optimizer.step()
        running_loss = running_loss / int(train_num_of_batches)
        logger.info('Epoch: {} , Running loss: {},Time {}\n'.format(epoch, running_loss, time.time() - start_time))
        if args.do_eval and args.evaluate_during_training:
            f1, _ = evaluate(args, model, eval_or_test_data, dev)
            if args.f1_threshold_value < f1:
                save_model_file_dir = os.path.join(args.output, f"checkpoint-{epoch}")
                if not os.path.exists(save_model_file_dir):
                    os.makedirs(save_model_file_dir)
                save_model_path = os.path.join(save_model_file_dir, "pytorch_model.bin")
                model.save_pretrained(save_model_file_dir)
                tokenizer.save_pretrained(save_model_file_dir)
                # Good practice: save your training arguments together with the trained model
                torch.save(model.state_dict(), save_model_path)
                logger.info(f"f1: {f1} Greater than the set value,save the model to {save_model_path}\n")


def predict():
    test_file_path = os.path.join(base_dir, "data", "eval_or_test_data.json")
    if os.path.exists(test_file_path):
        test_list = read_json(test_file_path)
        predict_tokenizer = MT5Tokenizer.from_pretrained(args.predict_model_path)
        predict_model = MT5ForConditionalGeneration.from_pretrained(args.predict_model_path, return_dict=True)
        generate(args, test_list, predict_tokenizer, predict_model)
    else:
        logger.info("No test data exists, please check !")


if __name__ == "__main__":
    if args.do_train:
        model_train()
    else:
        predict()
