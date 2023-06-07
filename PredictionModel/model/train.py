#from bert_ml_sentiment_classifier import bert_train, bert_evaluate
from ClassificationTrainer import ClassificationTrainer
from data_transformation import  prepare_data_for_training, prepare_data_for_testing
from utils import calculate_metrics

from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import torch
import pandas as pd
import numpy as np
import json

import random
import argparse
import os
#import csv
from math import floor

import math

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path",
                        required=False,
                        type=str)
    parser.add_argument("--eval_data_path",
                        required=False,
                        type=str)
    parser.add_argument("--test_data_path",
                        required=False,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--max_thread_len",
                        default=10,
                        type=int)
    parser.add_argument("--max_seg_len",
                        default=256,
                        type=int)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int)
    parser.add_argument("--num_epochs",
                        default=4,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float)
    parser.add_argument("--random_seed",
                        default=42,
                        type=int)
    parser.add_argument("--remember_checkpoints",
                        default=0,
                        type=int)
 
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("Setting the random seed...")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    print("Reading data and splitting into train/dev/test...")
    df_data = pd.read_csv("../data/dialog_dataset_for_tagger.csv")
    
    
    tax = {
                "InformationSeeking": ["Question","ClarificationElicitation"],
                "InformationProviding": ["Answer","Agreement","Disagreement","NeutralResponse","UnderstandingNegative", "ContextSetting", "ExtensionOfPrevious","ProposeAction","StateDecision"],
                "Social" : ["Thanking", "Apologising"]
    }
    inv_tax = {}
    for hlab in tax:
        for llab in tax[hlab]:
             inv_tax[llab] = hlab

    def expand_labels(l):
        labs = l.split(",")
        new_labs = []
        for lab in inv_tax:
            if lab in labs and inv_tax[lab] not in labs and inv_tax[lab] not in new_labs:
                new_labs.append(inv_tax[lab]) 
        return ",".join(labs + new_labs)
        

    df_data["labels"] = [expand_labels(x) for x in df_data.labels] 
   
    lab2weight = {}
    for lablist in df_data.labels:
      curlabs = lablist.split(",")
      for l in curlabs:
          if l not in lab2weight:
              lab2weight[l] = 0
          lab2weight[l] += 1
    lab_total_examples = df_data.shape[0]
    for l in lab2weight:
        lab2weight[l] = lab_total_examples / lab2weight[l]
        

    print(len(df_data))
    all_tids = list(set(df_data.thread_id))
    print(len(all_tids))
    random.seed(42)
    random.shuffle(all_tids)
    shuffled_all_tids =  all_tids

    dev_start_index = int(0.7 * len(shuffled_all_tids))
    test_start_index = int(0.85 * len(shuffled_all_tids))

    train_tids = set(shuffled_all_tids[:dev_start_index])
    dev_tids = set(shuffled_all_tids[dev_start_index: test_start_index])
    test_tids = set(shuffled_all_tids[test_start_index:])

    df_train_data = df_data[df_data.thread_id.isin(train_tids)]
    df_dev_data = df_data[df_data.thread_id.isin(dev_tids)]
    df_test_data = df_data[df_data.thread_id.isin(test_tids)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading pre-trained model...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertModel.from_pretrained('bert-base-cased')

    #tokenizer = BertTokenizer.from_pretrained('/scratch-local/ravi/codes/bert_tuning/output/ietf-bert-both', do_lower_case=False)
    #model = BertModel.from_pretrained('/scratch-local/ravi/codes/bert_tuning/output/ietf-bert-both')


    bert_trainer = ClassificationTrainer(model, device, tokenizer, batch_size=args.batch_size,
                                               lr=args.learning_rate, train_epochs=args.num_epochs,
                                               weight_decay=args.weight_decay,
                                               warmup_proportion=args.warmup_proportion,
                                               adam_epsilon=args.adam_epsilon,
                                               remember_checkpoints=args.remember_checkpoints,
                                               max_thread_len = args.max_thread_len,
                                               label_to_weight = lab2weight)

    
    #train_dataloader = prepare_data_for_training(df_train_data.sample(frac = 0.0001, replace = False), tokenizer, args.max_thread_len, args.max_seg_len, args.batch_size)
    train_dataloader = prepare_data_for_training(df_train_data, tokenizer, args.max_thread_len, args.max_seg_len, args.batch_size)
    eval_dataloader = prepare_data_for_training(df_dev_data, tokenizer, args.max_thread_len, args.max_seg_len, args.batch_size)
    test_dataloader = prepare_data_for_training(df_test_data, tokenizer, args.max_thread_len, args.max_seg_len, args.batch_size)

    print("Training...")
    bert_trainer.train(train_dataloader, eval_dataloader, output_dir, save_best=False, test_dataloader = test_dataloader)
    
    # test scores will be in log.txt

if __name__ == "__main__":
    run()
