"""
This file contains the model evaluation including NDCG, MAP, f1 score, ROC curve, precision vs. recall curve and statistical evaluation
"""

import json
import numpy as np
import pandas as pd
import seaborn as sn
import scipy.stats as st
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from config.config import CacheFile, EvalConfig
from sklearn.metrics import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from itertools import cycle


# cache
cache = CacheFile()
cache = CacheFile()
# data with 50 hashtags
X_TEST_50_PTH = cache.X_test_50
Y_TEST_50_PTH = cache.y_test_50
GT_VAL = cache.gt_val_50
GT_TEST = cache.gt_test_50
GT_STATS = cache.gt_stats


# evaluation
eval = EvalConfig()
NUM_SAMPLE = eval.n_sample()
TOP_N = [int(x) for x in eval.top_n().split(',')]



# prepare ground truth for evaluation
class GroundTruth:

    def __init__(self, dataset):
        self.dataset = dataset

    def get_ground_truth(self, save=False, save_pth=None):
        ground_truth = {i: [] for i in range(len(self.dataset))}
        for i in range(len(self.dataset)):
            hashtags = self.dataset[i][1]
            for j in range(len(hashtags)):
                if hashtags[j] == 1:
                    ground_truth[i].append(j)
        if save is True:
            try:
                json_ = json.dumps(ground_truth)
                with open(save_pth, 'w') as json_file:
                    json_file.write(json_)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")
        return ground_truth

    def get_stats_ground_truth(self, n_sample=NUM_SAMPLE, \
                               ground_truth_dict=None, \
                               save=False, save_pth=None):

        if ground_truth_dict is None:
            ground_truth_dict = self.get_ground_truth()

        # build new
        ground_truth = {i: [] for i in range(n_sample)}
        num = len(ground_truth_dict) - len(ground_truth_dict) % n_sample
        for i in list(ground_truth_dict.keys())[:num]:
            hash_value = int(i) % n_sample
            ground_truth[hash_value].append(ground_truth_dict[i])
        for i in range(n_sample):
            ground_truth[i] = {j: ground_truth[i][j] for j in range(len(ground_truth[i]))}

        if save is True:
            try:
                json_ = json.dumps(ground_truth)
                with open(save_pth, 'w') as json_file:
                    json_file.write(json_)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")
        return ground_truth
    
    def load_ground_truth(self, gt_pth):
        with open(gt_pth, 'r') as f:
            ground_truth = json.load(f)
        ground_truth = {int(k): v for k, v in ground_truth.items()}
        return ground_truth



# get predicted probability and (top n) recommended hashtags
class Predict:

    def __init__(self, val_dataloader, model, device, bert=False):
        self.data = val_dataloader
        self.model = model
        self.device = device
        self.bert = bert
       

    def get_pred_proba(self, save=False, save_pth=None):
        model = self.model.to(self.device)
        model.eval()

        pred_proba = torch.tensor([]).to(self.device)
        for i, data in enumerate(self.data):
            if self.bert == True:
                input_ids, attention_mask, token_type_ids, labels = \
                    data[0].to(self.device), data[1].to(self.device), \
                    data[2].to(self.device), data[3].to(self.device)
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            else:
                text, labels = data[0].to(self.device), data[1].to(self.device)
                out = model(text).to(self.device)
            pred_proba = torch.cat((pred_proba, out), 0)
        
        if save is True:
            try:
                np.save(save_pth, pred_proba.cpu().numpy())
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")

        return pred_proba


    def get_pred_top_k(self, top_n, pred_proba_val=None, save=False, save_pth=None):
        if pred_proba_val is None:
            pred_proba_val = self.get_pred_proba()

        pred_p_list = pred_proba_val.cpu().numpy().tolist()
        result = {i: [] for i in range(len(pred_p_list))}
        for i in range(len(pred_p_list)):
            hashtags_p = [[pred_p_list[i][j], j] for j in range(len(pred_p_list[i]))]
            hashtags_p.sort(key = lambda x: x[0], reverse = True)
            for n in range(top_n):
                result[i].append(hashtags_p[n][1])

        if save is True:
            try:
                json_ = json.dumps(result)
                with open(save_pth, 'w') as json_file:
                    json_file.write(json_)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")
        return result
    

    def get_pred_by_thre(self, threshold, pred_proba_val=None, save=False, save_pth=None):
        if pred_proba_val is None:
            pred_proba_val = self.get_pred_proba()

        pred_p_list = pred_proba_val.cpu().numpy().tolist()
        result = {i: [] for i in range(len(pred_p_list))}
        for i in range(len(pred_p_list)):
            hashtags_p = pred_p_list[i]
            max_p = 0
            for j in range(len(hashtags_p)):
                p = hashtags_p[j]
                if p >= threshold:
                    result[i].append(j)
                if p > max_p:
                    max_p = p
                    max_idx = j
                if not result[i]:
                    result[i].append(hashtags_p[max_idx])
        
        if save is True:
            try:
                json_ = json.dumps(result)
                with open(save_pth, 'w') as json_file:
                    json_file.write(json_)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")

        return result
    


# calculate metrics (NDCG and MAP)
class CalculateMetrics:
    def __init__(self, pred_dict, ground_truth):
        self.pred = pred_dict
        self.gt = ground_truth


    def _calculate_NDCG(self):
        ndcg = {i: 0 for i in self.pred.keys()}
        for key, val_list in self.pred.items():
            dcg = 0
            count = 0
            z = 0
            for i in range(len(val_list)):
                if type(list(self.gt.keys())[0]) == str:
                    key = str(key)
                if val_list[i] in self.gt[key]:
                    dcg += 1 / np.log2(1+i+1)
                    count += 1
                    z += 1 / np.log2(1+count)
            if dcg != 0:
                ndcg[int(key)] = dcg / z
        return sum(ndcg.values())/len(ndcg.values())


    def _calculate_MAP(self):
        precision = []
        for key, val_list in self.pred.items():
            pre = 0
            n_true = 0
            for i in range(len(val_list)):
                if val_list[i] in self.gt[str(key)]:
                    n_true += 1
                    pre += n_true / (i + 1)
            if n_true != 0:
                precision.append(pre / n_true)
        return sum(precision) / len(precision)
    

    def get_metrics(self):
        metric = []
        # ndcg
        ndcg = self._calculate_NDCG(self.pred, self.gt)
        metric.append(('NDCG', ndcg))
        # map
        map = self._calculate_MAP(self.pred, self.gt)
        metric.append(('MAP', map))
        return metric


# calculate metrics (classification report, confusion_matrix)
    def evaluation_report(ground_truth, pred_result):
        mb_ground_truth = MultiLabelBinarizer().fit_transform(list((ground_truth.values())))
        mb_pred_result = MultiLabelBinarizer().fit_transform(list((pred_result.values())))
        report = classification_report(mb_ground_truth,mb_pred_result)
        return report
    
    def confusion_matrix_sep(confusion_matrix, axes, class_label, class_names, fontsize = 10):
            df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
            try:
                heatmap = sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws= {"size": 10}, fmt="d", cbar=False, ax=axes)
            except ValueError:
                raise ValueError("Confusion matrix values must be integers.")
            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize= 5)
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize= 5)
            axes.set_ylabel('True label', fontsize = 10)
            axes.set_xlabel('Predicted label', fontsize = 10)
            axes.set_title("Confusion Matrix for the class - " + class_label, fontsize = 10)
     
    def plot_confusion_matrix_sep(ground_truth, pred_result):
        mb_ground_truth = MultiLabelBinarizer().fit_transform(list((ground_truth.values())))
        mb_pred_result = MultiLabelBinarizer().fit_transform(list((pred_result.values())))
        cm = multilabel_confusion_matrix(mb_ground_truth,mb_pred_result)
        labels = ["".join("c" + str(i)) for i in range(0, 50)]
        fig, ax = plt.subplots(10, 5, figsize=(30, 15))
        for axes, cfs_matrix, label in zip(ax.flatten(), cm, labels):
            confusion_matrix_sep(cfs_matrix, axes, label, ["N", "Y"])
        plt.show()
        
        
# calculate metrics (ROC curve, precision vs. recall curve)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 50
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(mb_y_test[:, i], pred_prob_tfidf[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(mb_y_test.ravel(), pred_prob_tfidf.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize = (20,15))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="darkred",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    colors = cycle(["aqua", "orange", "lightgreen", "lightpink", "gold","mediumpurple","chocolate"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            linewidth=4,
            label= "ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate",fontsize=20)
    plt.ylabel("True Positive Rate",fontsize=20)
    plt.title("ROC Curve",fontsize=20)
    plt.show()
    
    # Compute PR curve for each class
    precision = dict()
    recall = dict()
    n_classes = 50
    colors = cycle(["aqua", "orange", "lightgreen", "lightpink", "gold","mediumpurple","chocolate"])
    _, ax = plt.subplots(figsize=(20,15))
    for i, color in zip(range(n_classes), colors):
    precision[i], recall[i], _ = precision_recall_curve(mb_y_test[:, i], pred_prob_tfidf[:, i])
    plt.plot(recall[i], precision[i], color = color, lw=2, label='class {}'.format(i))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall",fontsize=20)
    plt.ylabel("Precision",fontsize=20)
    plt.title("Precision vs. Recall Curve",fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()



# statistical evaluation
def hash_func(X_pth=X_TEST_50_PTH, y_pth=Y_TEST_50_PTH, n_sample=NUM_SAMPLE):

    X = np.load(X_pth, allow_pickle=True)
    y = np.load(y_pth, allow_pickle=True)
    num = len(y) - len(y) % n_sample
    X = X.tolist()[:num]
    y = y.tolist()[:num]
    group = {i: {'text': [], 'labels': []} for i in range(50)}
    for i in range(len(y)):
        hash_value = i % n_sample
        group[hash_value]['text'].append(X[i])
        group[hash_value]['labels'].append(y[i])
    return group

# load Dataset
class TextDataset_stats(Dataset):
    def __init__(self, group_num, X_pth=X_TEST_50_PTH, 
                 y_pth=Y_TEST_50_PTH, n_sample=NUM_SAMPLE):
      """ 
      group_num: 0, 1, 2, ... , 49
      """
      self.data= hash_func(X_pth, y_pth, n_sample)[group_num]

    def __len__(self):
      x = self.data['labels']
      return len(x)

    def __getitem__(self, idx):
      text = self.data['text'][idx]
      labels = self.data['labels'][idx]
      return text, labels



class StatisticalEval:

    def __init__(self, ground_truth, device, n=NUM_SAMPLE, top_list=TOP_N, bert=False):
        self.gt = ground_truth
        self.device = device
        self.n = n
        self.bert = bert
        self.top_list = top_list
    
    def get_n_samples(self):
        dataset = TextDataset_stats(n_sample=self.n)
        return dataset
    
    def predict_group(self, model, collate_batch, dataset=None, save=False, save_pth=None):
        if dataset is None:
            dataset = self.get_n_samples(self.n)
        # build data loader in each group
        pred_result = dict()
        for m in range(self.n):
            group_dataloader = DataLoader(dataset=dataset(m),
                                    batch_size=256,
                                    collate_fn=collate_batch,
                                    shuffle=False,
                                    drop_last=False)
            # predict
            pred = Predict(group_dataloader, model, self.device, self.bert)

            # get top result
            pred_result[m] = {
                self.top_list[i]: pred.get_pred_top_k(self.top_list[i]) \
                for i in range(len(self.top_list))
                }

        if save is True:
            try:
                json_ = json.dumps(pred_result)
                with open(save_pth, 'w') as json_file:
                    json_file.write(json_)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")

        return pred_result
    

    def get_group_metrics(self, group_ground_truth, group_result=None, save=False, save_pth=None):

        metrics = []
        for i in range(self.n):
            m = []
            for j in self.top_list:
                res = group_result[i][j]
                truth = group_ground_truth[i]
                me = CalculateMetrics(res, truth)
                m.extend(me.get_metrics())
                metrics.append(m)
        columns = []
        for k in self.top_list:
            columns.extend(['NDCG@{}'.format(k), 'MAP@{}'.format(k)])
        df = pd.DataFrame(metrics, columns = columns)

        if save is True:
            try:
                df.to_csv(save_pth, index=False)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")

        return df
    

    def get_group_metrics_ci(self, df_metrics, sig_level=0.05, save=False, save_pth=None):

        df_ci = pd.DataFrame([])
        for col in df_metrics.columns:
            avg = df_metrics[col].mean()
            ci = st.t.interval(
                alpha=1-sig_level, 
                df=len(df_metrics)-1,
                loc=np.mean(df_metrics[col]),
                scale=st.sem(df_metrics[col])
                )
            df_ci[col] = [avg, ci]
            df = df_ci.T
            df.columns = ['mean', 'confidence interval']

        if save is True:
            try:
                df.to_csv(save_pth, index=True)
            except TypeError:
                raise FileNotFoundError("You must enter a file path when 'save=True'")

        return df
  
