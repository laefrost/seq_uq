from sklearn import metrics


def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    del thresholds
    return metrics.auc(fpr, tpr)


def prroc(y_true, y_score):
    ##precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score, drop_intermediate=True)
    #del thresholds
    return metrics.average_precision_score(y_true, y_score) #metrics.auc(recall, precision)