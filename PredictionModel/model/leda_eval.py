import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json
import sys

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
            'NeutralResponse':13,
            'ContextSetting':14, 
            'Disagreement':15, 
            'ClarificationElicitation':16}

colindex_to_label_map = {v: k for k, v in label_to_colindex_map.items()}

gold_file = sys.argv[1]
labels_file = sys.argv[2]

gold_json = json.load(open(gold_file, "r"))
labels_json = json.load(open(labels_file, "r"))

# SOME SANITY CHECKS SKIP LATER
goldids = [x["current_segment"]["sid"] for x in gold_json]
labids = [int(x) for x in list(labels_json.keys())]
labids = [x for x in labids if x!= -1]

print(len(gold_json))
print(len(labels_json))


print(goldids[0:50])
print(labids[0:50])

g,l,b = 0,0,0

for i in set(goldids + labids):
    if i in goldids and i in labids:
        b+=1
    if i in goldids and i not in labids:
        g+=1
    if i in labids and i not in goldids:
        l+=1

print(f"Both: {b}, Only gold: {g}, Only lab: {l}")

# EVALUATION
for lab in list(label_to_colindex_map.keys()):
    gold_labs, pred_labs = [], []
    for example in gold_json:
        sid = example["current_segment"]["sid"]
        gold_label = 1 if lab in example["current_segment"]["labels"] else 0
        pred_label = 1 if lab in labels_json[str(sid)] else 0
        gold_labs.append(gold_label)
        pred_labs.append(pred_label)
    p = precision_score(y_true = gold_labs, y_pred = pred_labs)
    r = recall_score(y_true = gold_labs, y_pred = pred_labs)
    f1 = f1_score(y_true = gold_labs, y_pred = pred_labs)

    print("P = %.3f, R = %.3f, F1 = %.3f <-- %s" % (p, r, f1, lab))






