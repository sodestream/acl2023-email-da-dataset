import torch
#from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
#from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import os
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#from sklearn.model_selection import train_test_split
#from preprocess import transform_data
    
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



class ClassificationTrainer():
    """Finetunes a transformer-based model on a classification task"""

    def __init__(self, model, device, tokenizer, batch_size=32, lr=2e-5, train_epochs=3, weight_decay=0.01,
                 warmup_proportion=0.1, adam_epsilon=1e-8, remember_checkpoints=0, max_thread_len=0, label_to_weight = None):
        self.device = device
        self.model = model
        self.bilstm_layer = torch.nn.LSTM(input_size = 768, hidden_size = 256, batch_first = True, bidirectional = True)
        self.linear_layer = torch.nn.Linear(512, 17)
        self.transfer_func_layer = torch.nn.Sigmoid()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.train_epochs = train_epochs
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion
        self.adam_epsilon = adam_epsilon
        self.remember_checkpoints = remember_checkpoints
        self.max_thread_len = max_thread_len
        self.index_to_label_map = {}
        self.index_to_label_weight_map = {}
        for k in label_to_colindex_map:
            self.index_to_label_map[label_to_colindex_map[k]] = k
            self.index_to_label_weight_map[label_to_colindex_map[k]] = label_to_weight[k]

    def train(self, train_dataloader, eval_dataloader, output_dir, save_best=False, eval_metric='f1', test_dataloader = None):
        """Training loop for classification fine-tuning."""

        t_total = len(train_dataloader) * self.train_epochs
        warmup_steps = len(train_dataloader) * self.warmup_proportion
        no_decay = ['bias', 'LayerNorm.weight']
        #for p in self.bilstm_layer.named_parameters():
        #  print(p)
       
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.bilstm_layer.named_parameters()],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.linear_layer.named_parameters()],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        train_iterator = trange(int(self.train_epochs), desc="Epoch")
        #model = self.model
        self.model.to(self.device)
        self.bilstm_layer.to(self.device)
        self.linear_layer.to(self.device)
        self.transfer_func_layer.to(self.device)
        tr_loss_track = []
        eval_metric_track = []
        test_metric_track = []
        output_filename = os.path.join(output_dir, 'pytorch_model.bin')
        metric = float('-inf')

        for epoch_num in train_iterator:
            self.model.train()
            self.model.zero_grad()
            self.bilstm_layer.train()
            self.bilstm_layer.zero_grad()
            self.linear_layer.train()
            self.linear_layer.zero_grad()
            tr_loss = 0
            nr_batches = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                
                input_ids, input_mask, labels, segment_ids = batch
                tr_loss += self.__train_step(input_ids, input_mask, labels, optimizer, scheduler)
                nr_batches += 1
                self.model.zero_grad()
                self.bilstm_layer.zero_grad()
                self.linear_layer.zero_grad()
                #if step > 20:
                #    break

            print("Evaluating the model on the evaluation split...")
            metrics = self.evaluate(eval_dataloader)
            eval_metric_track.append(metrics["out_str"])

            test_metrics = self.evaluate(test_dataloader)
            test_metric_track.append(test_metrics["out_str"])

            print("-------- EVALUATING AFTER EPOCH ----------------")
            print("Train loss function is " + str(tr_loss))
            print("EVAL SET:")
            print(metrics["out_str"])
            print(metrics[eval_metric])
            print("TEST SET:")
            print(test_metrics[eval_metric])
            print(test_metrics["out_str"])
            print("------------------------------------------------")

            # rewrite all results to log
            with open("log.txt", "w") as outfile:
                for i in range(len(eval_metric_track)):
                    outfile.write(eval_metric_track[i] + "\n")
                    outfile.write(test_metric_track[i] + "\n")
                    outfile.write(str(metrics[eval_metric]) + "\n")
                    outfile.write(str(test_metrics[eval_metric]) + "\n")
                    
                    outfile.write("\n-------------------------------------------\n")
            
        
            if save_best:
                if metric < metrics[eval_metric]:
                    self.model.save_pretrained(output_dir)
                    torch.save(self.model.state_dict(), output_filename)
                    print("The new value of " + eval_metric + " score of " + str(metrics[eval_metric]) + " is higher then the old value of " +
                          str(metric) + ".")
                    print("Saving the new model...")
                    metric = metrics[eval_metric]
                else:
                    print(
                        "The new value of " + eval_metric + " score of " + str(metrics[eval_metric]) + " is not higher then the old value of " +
                        str(metric) + ".")

            if self.remember_checkpoints == 1:
                checkpoint_output_filename = os.path.join(output_dir, 'checkpoint_epoch-%d-pytorch_model.bin' % (epoch_num))
                torch.save(self.model.state_dict(), checkpoint_output_filename)

            tr_loss = tr_loss / nr_batches
            tr_loss_track.append(tr_loss)
        
        if not save_best:
            self.model.save_pretrained(output_dir)
            # tokenizer.save_pretrained(output_dir)
            torch.save(self.model.state_dict(), output_filename)

        
        return tr_loss_track, eval_metric_track

    def evaluate(self, eval_dataloader):
        """Evaluation of trained checkpoint."""
        
        self.model.to(self.device)
        self.bilstm_layer.to(self.device)
        self.linear_layer.to(self.device)

        self.model.eval()
        self.bilstm_layer.eval()
        self.linear_layer.eval()

        predictions = {}
        true_labels = {}

        data_iterator = tqdm(eval_dataloader, desc="Iteration")
        sid2prediction = {}
        sid2trueval = {}
 
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask, labels, segment_ids = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                # (B*Mseg) X  Mtok → (Encoder) → (B*Mseg) X E → (reshape) → B X Mseg X E → (BiLSTM) → B X Mseg X H → (linear) → B X Mseg X C → sigmoid for each output → B X Mseg X C
                #print(input_ids.shape)
                #print(input_mask.shape)
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=input_mask).pooler_output # (B * max_thread_len) X embedding_size
                #print(outputs.shape)
                # reshape to batch of segment sequences
                outputs_split = torch.split(outputs, self.max_thread_len, dim = 0) # list of  max_thread_len X embedding_size chunks (B of them in total)
                lstm_input = torch.stack(outputs_split, dim = 0) # now we have B X max_thread_len X embedding_size
                #print(lstm_input.shape)

                input_mask_per_segment = torch.sum(input_mask, dim = 1)
                #print("mask per seg shape")
                #print(input_mask_per_segment.shape)
                #print("shapes after some more transformations")
                input_mask_per_segment_split = torch.split(input_mask_per_segment, self.max_thread_len, dim = 0)
                input_mask_per_segment_for_lstm_loss = torch.stack(input_mask_per_segment_split, dim = 0) # B X max_thread_len
                #print(input_mask_per_segment_for_lstm_loss.shape)


                labels_split = torch.split(labels, self.max_thread_len, dim = 0)
                labels_for_loss = torch.stack(labels_split, dim = 0) # B X max_thread_len X n_labels
                #print(labels_for_loss.shape)
                
                segment_ids_split = torch.split(segment_ids, self.max_thread_len, dim = 0)
                segment_ids_for_loss = torch.stack(segment_ids_split, dim = 0)

                # run BiLSTM over each segment sequence in batch
                lstm_output, (hn, cn) = self.bilstm_layer(lstm_input) # lstm_output is B X max_thread_len X hidden_size 
                #print(lstm_output.shape)
                # run Linear layer over each BiLSTM cell/hidden state
                linear_output = self.linear_layer(lstm_output) # B x max_thread_len X hidden_size multiplied by hidden_size X n_labels --> B X max_thread_len X n_labels
                # sigmoid the outputs 
                #print(linear_output.shape)
                logits = self.transfer_func_layer(linear_output)
                # change the loss computation
                #print("Final output")
                #print(final_output.shape)
            #print(logits.shape)
            for batch_index in range(logits.shape[0]):
                for segment_index in range(logits.shape[1]):
                   if input_mask_per_segment_for_lstm_loss[batch_index, segment_index].item() > 0:
                     for label_index in range(logits.shape[2]):               
                        #print("_________")
                        #print(batch_index)         
                        #print(segment_index)         
                        #print(label_index)         
                        #print(logits[batch_index, segment_index, label_index])
                        predicted_value = 1 if logits[batch_index, segment_index, label_index].item() >= 0.5 else 0
                        true_value = int(labels_for_loss[batch_index, segment_index, label_index].item())
                        if label_index not in sid2prediction:
                            sid2prediction[label_index] = {}
                            sid2trueval[label_index] = {}
                        current_segment_id = int(segment_ids_for_loss[batch_index, segment_index].item())
                        if current_segment_id not in sid2prediction[label_index]:
                            sid2prediction[label_index][current_segment_id] = []
                        if current_segment_id not in sid2trueval[label_index]:
                            sid2trueval[label_index][current_segment_id] = []
                        sid2prediction[label_index][current_segment_id].append(predicted_value)
                        sid2trueval[label_index][current_segment_id].append(true_value)

        f1_list, precision_list, recall_list, f1binary_list = [], [], [], []
        output_string = ""
        for label_index in range(17):
            current_true, current_preds = [], []
            
            everything_ok = True
            for sid in set(list(sid2prediction[label_index].keys()) + list(sid2trueval[label_index].keys())):
                if sid not in sid2trueval[label_index]:
                        everything_ok = False
                if sid not in sid2prediction[label_index]:
                        everything_ok = False

            assert everything_ok

            for sid in sid2prediction[label_index]:
                true_list = sid2trueval[label_index][sid]
                #print(true_list)
                assert len(set(true_list)) == 1
                aggregated_true = true_list[0] # all elements should be the same in the gold standard
                current_true.append(aggregated_true)

                pred_list = sid2prediction[label_index][sid]
                aggregated_prediction = max(set(pred_list), key = pred_list.count) # most frequent prediction for that segment

                current_preds.append(aggregated_prediction)
             

            print("status for label" + str(self.index_to_label_map[label_index]))
            #print(current_true)
            #print(current_preds)
            score = f1_score(y_true = current_true, y_pred = current_preds, pos_label = 1, average = "macro", zero_division = 0)
            precision = precision_score(y_true = current_true, y_pred = current_preds, pos_label = 1, average = "macro", zero_division = 0)
            recall = recall_score(y_true = current_true, y_pred = current_preds, pos_label = 1, average = "macro", zero_division = 0)

            score_binary = f1_score(y_true = current_true, y_pred = current_preds, pos_label = 1, average = "binary", zero_division = 0)
            precision_binary = precision_score(y_true = current_true, y_pred = current_preds, pos_label = 1, average = "binary", zero_division = 0)
            recall_binary = recall_score(y_true = current_true, y_pred = current_preds, pos_label = 1, average = "binary", zero_division = 0)

            f1_list.append(score)
            precision_list.append(precision)
            recall_list.append(recall)
            f1binary_list.append(score_binary)

            print("CLASS %s score is %.4f (precision is %.4f and recall is %.4f" % (self.index_to_label_map[label_index],score, precision,recall))
            print("(binary averaging) CLASS %s score is %.4f (precision is %.4f and recall is %.4f" % (self.index_to_label_map[label_index],score_binary, precision_binary,recall_binary))
            output_string += " %20s -> Macro: %.3f / %.3f /%.3f Bin: %.3f/ %.3f / %.3f,  " % (self.index_to_label_map[label_index], score, precision, recall,  score_binary, precision_binary, recall_binary)
        print("----------------------")

        metrics = {
                    "f1": np.mean(f1_list),
                    "out_str": output_string
                  }
        return metrics

    def predict(self, predict_dataloader, return_probabilities=False):
        """Testing of trained checkpoint.
        CHANGE SO IT DOES NOT TAKE DATALOADER WITH LABELS BECAUSE WE DON'T NEED THEM"""
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        probabilities = []
        # true_labels = []
        data_iterator = tqdm(predict_dataloader, desc="Iteration")
        softmax = torch.nn.Softmax(dim=-1)
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)

            # loss is only output when labels are provided as input to the model ... real smooth
            logits = outputs[0]
            #print(type(logits))
            probs = softmax(logits)
            logits = logits.to('cpu').numpy()
            probs = probs.to('cpu').numpy()
            # label_ids = labels.to('cpu').numpy()

            for l, prob in zip(logits, probs):
                # true_labels.append(label)
                predictions.append(np.argmax(l))
                probabilities.append(prob)

        # print(predictions)
        # print(true_labels)
        # metrics = get_metrics(true_labels, predictions)
        if return_probabilities == False:
            return predictions
        else:
            return predictions, probabilities

    def __train_step(self, input_ids, input_mask, labels, optimizer, scheduler):
        positive_weights = []
        for k in sorted(self.index_to_label_weight_map.keys()):
           positive_weights.append(self.index_to_label_weight_map[k])
        positive_class_weights = torch.tensor(positive_weights).to(self.device)
        #print(positive_class_weights)
        #exit()
  
        # (B*Mseg) X  Mtok → (Encoder) → (B*Mseg) X E → (reshape) → B X Mseg X E → (BiLSTM) → B X Mseg X H → (linear) → B X Mseg X C → sigmoid for each output → B X Mseg X C
        #print(input_ids.shape)
        #print(input_mask.shape)
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        labels = labels.to(self.device)
        optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask=input_mask).pooler_output # (B * max_thread_len) X embedding_size
        #print(outputs.shape)
        # reshape to batch of segment sequences
        outputs_split = torch.split(outputs, self.max_thread_len, dim = 0) # list of  max_thread_len X embedding_size chunks (B of them in total)
        lstm_input = torch.stack(outputs_split, dim = 0) # now we have B X max_thread_len X embedding_size
        #print(lstm_input.shape)

        input_mask_per_segment = torch.sum(input_mask, dim = 1)
        #print("mask per seg shape")
        #print(input_mask_per_segment.shape)
        #print("shapes after some more transformations")
        input_mask_per_segment_split = torch.split(input_mask_per_segment, self.max_thread_len, dim = 0)
        input_mask_per_segment_for_lstm_loss = torch.stack(input_mask_per_segment_split, dim = 0) # B X max_thread_len
        #print(input_mask_per_segment_for_lstm_loss.shape)


        labels_split = torch.split(labels, self.max_thread_len, dim = 0)
        labels_for_loss = torch.stack(labels_split, dim = 0) # B X max_thread_len X n_labels
        #print(labels_for_loss.shape)

        # run BiLSTM over each segment sequence in batch
        lstm_output, (hn, cn) = self.bilstm_layer(lstm_input) # lstm_output is B X max_thread_len X hidden_size 
        #print(lstm_output.shape)
        # run Linear layer over each BiLSTM cell/hidden state
        linear_output = self.linear_layer(lstm_output) # B x max_thread_len X hidden_size multiplied by hidden_size X n_labels --> B X max_thread_len X n_labels
        # ***************************************************************************
        # sigmoid the outputs 
        #print(linear_output.shape)
        ######################final_output = self.transfer_func_layer(linear_output)
        #print(final_output.shape)
        # change the loss computation
        #print("Final output")
        #print(final_output.shape)
        
        ####################loss = torch.nn.BCELoss(reduction = 'none')(final_output.float(), labels_for_loss.float()) # B X max_thread_len X n_labels
        #****************************************************************************

        loss = torch.nn.BCEWithLogitsLoss(reduction = 'none', pos_weight = positive_class_weights)(linear_output.float(), labels_for_loss.float())
        loss_averaged_over_labels = torch.mean(loss, dim = 2)
        loss_masked = torch.mul(loss_averaged_over_labels, input_mask_per_segment_for_lstm_loss)
        #print("Loss masked")
        #print(loss_masked.shape)
        final_loss = torch.mean(loss_masked)
        #print("Final loss")
        #print(final_loss.shape)
        #print(final_loss)
        final_loss.backward()
        optimizer.step()
        scheduler.step()
        return final_loss.float()
   
 
