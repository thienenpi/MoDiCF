import utils.metrics as metrics
import multiprocessing
import heapq
import numpy as np
import torch
import re
from utils.metrics_eval import metrics_dict

cores = multiprocessing.cpu_count() // 2

class MRS_test:
    def __init__(self, data, batch_size, test_flag, Ks, incomplete_item_num=0):
        self.data = data
        self.batch_size = batch_size
        self.USR_NUM, self.ITEM_NUM = self.data.n_users, self.data.n_items
        self.N_TRAIN, self.N_TEST = self.data.n_train, self.data.n_test
        self.test_flag = test_flag
        self.Ks = Ks
        self.INCOMPLETE_ITEM_NUM = incomplete_item_num
        self.p_incomplete_dataset = incomplete_item_num / self.ITEM_NUM


    def test_torch_counterfactual(self, ua_embeddings, ia_embeddings, IRS_score_sigmoid,
                                  users_to_test, is_val, incomplete_items=None, c=40, export=False,
                                  debias_mode=None, gamma_train=0.0,
                                  user_group=None, group_margin=None, lambda_cf=1.0):
        metric_list = ['recall', 'precision', 'hit_ratio', 'ndcg', 'fair_incomplete', 'fair_p', 'f_fair']
        test_users = users_to_test
        test_items = range(self.ITEM_NUM)
        n_test_users = len(test_users)
        ua_embeddings = ua_embeddings[test_users, :].detach().cpu().numpy()
        ia_embeddings = ia_embeddings[test_items, :].detach().cpu().numpy()
        ratings = np.matmul(ua_embeddings, ia_embeddings.T)
        IRS_score_sigmoid = IRS_score_sigmoid.detach().cpu().numpy()
        if is_val:
            gt = self.data.val_set
            pos_len_list = self.data.val_len_list
        else:
            gt = self.data.test_set
            pos_len_list = self.data.test_len_list

        if incomplete_items is not None:
            missing_items = list(incomplete_items.keys())
        else:
            missing_items = None

        mask = self.data.pos_items_per_u

        result_dic = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)),
                  'ndcg': np.zeros(len(self.Ks)),
                  'hit_ratio': np.zeros(len(self.Ks)),
                  'fair_incomplete': np.zeros(len(self.Ks)), 'fair_p': np.zeros(len(self.Ks)),
                  'f_fair': np.zeros(len(self.Ks))}
        if debias_mode is not None:
            # Training-time deconfounding: inference rule MUST match the training rule.
            if debias_mode == "subtract" and user_group is not None:
                # Group margin: users sharing a "hobby" cluster share one gamma_{g,i}.
                margin = group_margin[user_group[test_users]]  # [n_test_users, n_items]
                rate_ori = ratings - lambda_cf * margin * IRS_score_sigmoid[test_items]
            elif debias_mode == "subtract":      # train+test on (u.i - gamma*sigm(y_i))
                rate_ori = ratings - gamma_train * IRS_score_sigmoid[test_items]
            elif debias_mode == "ipw":          # IPW reweights training only; serve plain relevance
                rate_ori = ratings.copy()
            elif debias_mode == "multiply":     # MoDiCF Eq.17 rule (baseline / isolation run)
                rate_ori = (ratings - c) * IRS_score_sigmoid[test_items]
            else:
                raise ValueError("unknown debias_mode %r" % debias_mode)
        else:
            # Eq. 17: y_new = (y_{u,i} - gamma) * sigm(y_i); here c == gamma (constant).
            rate_ori = (ratings - c) * IRS_score_sigmoid[test_items]

        rate = rate_ori.copy()
        rate[mask[0, :], mask[1, :]] = -np.inf

        _, topk_index = torch.topk(torch.tensor(rate), k=max(self.Ks), dim=-1)
        topk_index = topk_index.numpy()

        bool_rec_matrix = []
        bool_incomplete_matrix = []

        if missing_items is not None:
            for i in range(n_test_users):
                r = topk_index[i, :]
                pos_items = gt[test_users[i]]
                bool_rec_matrix.append(np.isin(r, pos_items))
                bool_incomplete_matrix.append(np.isin(r, missing_items))
                if export:
                    for j in range(len(r)):
                        if r[j] in missing_items:
                            incomplete_items[r[j]] += 1
            bool_incomplete_matrix = np.array(bool_incomplete_matrix)
        else:
            for i in range(n_test_users):
                r = topk_index[i, :]
                pos_items = gt[test_users[i]]
                bool_rec_matrix.append(np.isin(r, pos_items))
            bool_incomplete_matrix = None
        bool_rec_matrix = np.array(bool_rec_matrix)

        result_list = []
        for metric in metric_list:
            if metric == 'fair_incomplete':
                if bool_incomplete_matrix is None:
                    raise ValueError('The topk_incomplete is None, but you use fair_incomplete metric')
                metric_fuc = metrics_dict['fair_incomplete']
                result = metric_fuc(bool_incomplete_matrix)
                result_list.append(result)
                continue
            if metric == 'fair_p':
                if self.p_incomplete_dataset is None:
                    raise ValueError('The p_incomplete_d is None, but you use fair_p metric')
                metric_fuc = metrics_dict['fair_p']
                result = metric_fuc(bool_incomplete_matrix, self.p_incomplete_dataset)
                result_list.append(result)
                continue
            if metric == 'f_fair':
                if bool_incomplete_matrix is None or self.p_incomplete_dataset is None:
                    raise ValueError('The topk_incomplete or p_incomplete_d is None, but you use f_fair metric')
                fair_p_function = metrics_dict['fair_p']
                precision_function = metrics_dict['precision']
                fair = fair_p_function(bool_incomplete_matrix, self.p_incomplete_dataset)
                precision = precision_function(bool_rec_matrix, pos_len_list)
                metric_fuc = metrics_dict['f_fair']
                result = metric_fuc(fair, precision)
                result_list.append(result)
                continue
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(bool_rec_matrix, pos_len_list)
            result_list.append(result)
        for m, v in zip(metric_list, result_list):
            for k_idx in range(len(self.Ks)):
                k = self.Ks[k_idx]
                result_dic[m][k_idx] = v[k - 1]

        result_dic['c'] = c
        return result_dic
