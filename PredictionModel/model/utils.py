from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

def calculate_metrics(labels, predictions):
    metrics = {}
    metrics['cm'] = confusion_matrix(labels, predictions)
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['cm'] = metrics['cm'].tolist()
    metrics['macro_precision'] = precision_score(labels, predictions, average='macro')
    metrics['macro_recall'] = recall_score(labels, predictions, average='macro')

    f1_scores = f1_score(labels, predictions, average=None)
    for i, f in enumerate(f1_scores):
        name = "F1_" + str(i)
        metrics[name] = f
    metrics['F1_macro'] = f1_score(labels, predictions, average='macro')
    return metrics