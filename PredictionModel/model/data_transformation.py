import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import numpy as np

#from keras_preprocessing.sequence import pad_sequences

#import math
import sys

def encode_labels_as_binary(labels):
    label_to_colindex_map = {   
            'Question':0, 
            'InformationProviding':1, 
            'Answer':2, 
            'Social':3, 
            'ExtensionOfPrevious':4, 
            'StateDecision':5, 
            'Thanking':6, 
            'ProposeAction':7, 
            'InformationSeeking':8, 
            'Agreement':9, 
            'Apologising':10, 
            'UnderstandingNegative':11, 
            'FollowOnComment':12, 
            'NeutralResponse':13,
            'ContextSetting':14, 
            'Disagreement':15, 
            'ClarificationElicitation':16}

    encoded_labels = np.zeros((len(labels), 17))
    for row, lab_list in enumerate(labels):
      if "," not in lab_list:
          continue
      for lab in lab_list.split(","):
          encoded_labels[row, label_to_colindex_map[lab]] = 1

    return encoded_labels


def prepare_data_for_training(input_df, tokenizer, max_path_len, max_seg_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    #sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]
    
    # prepare for each path a sequence of segments padded to max_path_len
    data, labels, sids =[], [], []
    total_paths, modified_paths = 0, 0
    for tid in set(input_df.thread_id):
        thread_df = input_df[input_df.thread_id == tid]
        for pid in set(thread_df.path_id):
            path_df = thread_df[thread_df.path_id == pid]
            sorted_path_df = path_df.sort_values(by = ["position_in_path"])
            # actual data
            path_text_data = list(sorted_path_df.content)
            path_label_data = list(sorted_path_df.labels)
            path_sid_data = [int(x) for x in list(sorted_path_df.unique_segment_id)]

            # cutoff paths that are too long if needed  
            if len(path_text_data) > max_path_len:
                modified_paths += 1
            total_paths += 1
            path_text_data = path_text_data[0: max_path_len]
            path_label_data = path_label_data[0: max_path_len]
            path_sid_data = path_sid_data[0: max_path_len]

            # pad paths that are too short if needed
            if max_path_len > len(path_text_data):
                path_text_data += [""] * (max_path_len - len(path_text_data))
                path_label_data += [""] * (max_path_len - len(path_label_data))
                path_sid_data += [-1] * (max_path_len - len(path_sid_data))

            # remember this in the total data
            data += path_text_data
            labels += path_label_data
            sids += path_sid_data

    print(len(data))
    print(len(labels))
    print(modified_paths,total_paths, modified_paths / total_paths)
    tokenized_segments = [tokenizer.tokenize(segment) for segment in data]
    truncated_segments = [segment[:(max_seg_len - 2)] for segment in tokenized_segments]
    truncated_segments = [["[CLS]"] + segment + ["[SEP]"] for segment in truncated_segments]
    print("Example of tokenized segment:")
    print(truncated_segments[0])

    input_ids = [tokenizer.convert_tokens_to_ids(segment) for segment in truncated_segments]
    print("Printing encoded segments:")
    print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids_padded = []
    for i in input_ids:
        while len(i) < max_seg_len:
            i.append(0)
        #print(len(i))
        input_ids_padded.append(i)


    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids_padded)
    labels = torch.tensor(encode_labels_as_binary(labels))
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(sids)

    print(input_ids.shape)
    print(labels.shape)
    print(attention_masks.shape)
    print(segment_ids.shape)

    transformed_data = TensorDataset(input_ids, attention_masks, labels, segment_ids)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, batch_size=batch_size * max_path_len)

    return dataloader


def prepare_data_for_testing(data, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    #sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]
    
    # prepare for each path a sequence of segments padded to max_path_len
    data =[], []
    for tid in set(input_df.thread_id):
        thread_df = input_df[input_df.thread_id == tid]
        for pid in set(thread_df.path_id):
            path_df = thread_df[thread_df.path_id == pid]
            sorted_path_df = path_df.sort_values(by = ["position_in_path"])
            # actual data
            path_text_data = list(sorted_path_df.content)
            # cutoff paths that are too long if needed
            path_text_data = path_text_data[0: max_path_len]
            # pad paths that are too short if needed
            if max_path_len > len(path_text_data):
                path_text_data += [""] * (max_path_len - len(path_text_data))
         
            # remember this in the total data
            data += path_text_data


    tokenized_segments = [tokenizer.tokenize(segment) for segment in data]
    truncated_segments = [segment[:(max_seg_len - 2)] for segment in tokenized_segments]
    truncated_segments = [["[CLS]"] + segment + ["[SEP]"] for segment in truncated_segments]
    print("Example of tokenized segment:")
    print(truncated_segments[0])

    input_ids = [tokenizer.convert_tokens_to_ids(segment) for segment in truncated_segments]
    print("Printing encoded segments:")
    print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids_padded = []
    for i in input_ids:
        while len(i) < max_seg_len:
            i.append(0)
        #print(len(i))
        input_ids_padded.append(i)


    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids_padded)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader
