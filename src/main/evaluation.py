
import numpy as np

def get_pred_result_top(pred_proba_val, top_n):
  pred_p_list = pred_proba_val.cpu().numpy().tolist()
  result = {i: [] for i in range(len(pred_p_list))}
  for i in range(len(pred_p_list)):
    hashtags_p = [[pred_p_list[i][j], j] for j in range(len(pred_p_list[i]))]
    hashtags_p.sort(key = lambda x: x[0], reverse = True)
    for n in range(top_n):
      result[i].append(hashtags_p[n][1])
  return result

# def get_pred_result(pred_proba_val, threshold):
#   pred_p_list = pred_proba_val.cpu().numpy().tolist()
#   result = {i: [] for i in range(len(pred_p_list))}
#   for i in range(len(pred_p_list)):
#     hashtags_p = pred_p_list[i]
#     max_p = 0
#     for j in range(len(hashtags_p)):
#       p = hashtags_p[j]
#       if p >= threshold:
#         result[i].append(j)
#       if p > max_p:
#         max_p = p
#         max_idx = j
#     if not result[i]:
#       result[i].append(hashtags_p[max_idx])
#   return result
  

def calculate_NDCG(top_10_dic, ground_truth):
    ndcg = {i:0 for i in top_10_dic.keys()}
    for key, val_list in top_10_dic.items():
      dcg = 0
      count = 0
      z = 0
      for i in range(len(val_list)):
          if val_list[i] in ground_truth[str(key)]:
              dcg += 1 / np.log2(1+i+1)
              count += 1
              z += 1 / np.log2(1+count)
      if dcg != 0:
          ndcg[key] = dcg / z

    return ndcg


def calculate_MAP(top_10_dic, ground_truth):
    precision = []
    for key, val_list in top_10_dic.items():
        pre = 0
        n_true = 0
        for i in range(len(val_list)):
            if val_list[i] in ground_truth[str(key)]:
                n_true += 1
                pre += n_true / (i + 1)
        if n_true != 0:
            precision.append(pre / n_true)
    return sum(precision) / len(precision)