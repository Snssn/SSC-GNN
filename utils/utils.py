import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from collections import defaultdict

from dgl.data.utils import load_graphs
import os

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
import cv2

"""
    Utility functions to handle data and evaluate model.
"""
def adjlist_to_sparse_matrix(adj_list, num_nodes):
        row, col = [], []
        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                row.append(node)
                col.append(neighbor)
        data = np.ones(len(row), dtype=np.float32)  
        adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        return adj_matrix

def load_data(data, prefix="data/"):

    if data == "yelp":
        data_file = loadmat(prefix + "YelpChi.mat")
        labels = data_file["label"].flatten()
        feat_data = data_file["features"].todense().A
        # load the preprocessed adj_lists
        with open(prefix + "yelp_homo_adjlists.pickle", "rb") as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + "yelp_rur_adjlists.pickle", "rb") as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + "yelp_rtr_adjlists.pickle", "rb") as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + "yelp_rsr_adjlists.pickle", "rb") as file:
            relation3 = pickle.load(file)
        file.close()
    elif data == "amazon":
        data_file = loadmat(prefix + "Amazon.mat")
        labels = data_file["label"].flatten()
        feat_data = data_file["features"].todense().A
        # load the preprocessed adj_lists
        with open(prefix + "amz_homo_adjlists.pickle", "rb") as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + "amz_upu_adjlists.pickle", "rb") as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + "amz_usu_adjlists.pickle", "rb") as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + "amz_uvu_adjlists.pickle", "rb") as file:
            relation3 = pickle.load(file)
    elif data == "tfinance":
        data_file = os.path.join(prefix, "tfinance")
        graph_tfinance, _ = load_graphs(data_file)
        graph_tfinance = graph_tfinance[0]
        labels = graph_tfinance.ndata["label"].argmax(1)
        labels = np.array(labels.cpu())
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            homo = pickle.load(file)
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            relation1 = pickle.load(file)
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            relation2 = pickle.load(file)
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            relation3 = pickle.load(file)
        feat_data = np.array(graph_tfinance.ndata["feature"])

    return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
    """
    Row-normalize sparse matrix
    Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, "wb") as file:
        pickle.dump(adj_lists, file)
    file.close()


def pos_neg_split(nodes, labels):
    """
    Split nodes into positive and negative given their labels.
    :param nodes: A list of nodes.
    :param labels: A list of node labels.
    :returns: A tuple of two lists, containing the positive and negative nodes respectively.
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes) 
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
    """
    Test the performance of GraphSAGE
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    """

    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)

        gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    f1_binary_1_gnn = f1_score(
        labels, np.array(gnn_pred_list), pos_label=1, average="binary"
    )
    f1_binary_0_gnn = f1_score(
        labels, np.array(gnn_pred_list), pos_label=0, average="binary"
    )
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average="micro")
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average="macro")
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(
        f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"
        + f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}"
    )
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def prob2pred(y_prob, thres=0.5):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(y_prob, dtype=np.int32)
    y_pred[y_prob >= thres] = 1
    y_pred[y_prob < thres] = 0
    return y_pred


class rate_res:
    def __init__(self):
        self.rate = 0
        self.rate_f = 0
        self.rate_b = 0
        self.total_f = 0
        self.total_b = 0


def test_dig(test_cases, labels, model, batch_size, thres=0.5, rl_has_trained=True):
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    f1_label1 = 0.0
    acc_label1 = 0.00
    recall_label1 = 0.0
    gnn_pred_list = []
    gnn_prob_list = []

    r1 = rate_res()
    r2 = rate_res()
    r3 = rate_res()
    rate = [r1, r2, r3]

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob, label_prob1, rate_list = model.to_prob(
            batch_nodes,
            batch_label,
            train_flag=False,
            rl_train_flag=False,
            rl_has_trained=rl_has_trained,
        )

        gnn_prob_arr = gnn_prob.detach().data.cpu().numpy()[:, 1]
        # gnn_pred = prob2pred(gnn_prob_arr, thres)
        # gnn_prob has been applied softmax, let us use the argmax
        gnn_pred = gnn_prob.detach().data.cpu().numpy().argmax(axis=1)
        f1_label1 += 0
        acc_label1 += 0
        recall_label1 += 0

        if len(rate_list) == 3 and len(rate_list[0]) > 0:
            for i in range(len(rate_list)):
                rate[i].rate += rate_list[i][0]
                rate[i].rate_f += rate_list[i][1]
                rate[i].rate_b += rate_list[i][2]
                rate[i].total_f += rate_list[i][3]
                rate[i].total_b += rate_list[i][4]

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())
        # label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

    for i in range(len(rate_list)):
        rate[i].rate /= test_batch_num
        rate[i].rate_f /= test_batch_num
        rate[i].rate_b /= test_batch_num

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_prob_list))
    auc_label1 = 0
    ap_label1 = 0

    f1_binary_1_gnn = f1_score(
        labels, np.array(gnn_pred_list), pos_label=1, average="binary"
    )
    f1_binary_0_gnn = f1_score(
        labels, np.array(gnn_pred_list), pos_label=0, average="binary"
    )
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average="micro")
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average="macro")
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(
        f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"
        + f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}"
    )
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    print(
        f"Label1 F1: {f1_label1 / test_batch_num:.4f}\tAccuracy: {acc_label1 / test_batch_num:.4f}"
        + f"\tRecall: {recall_label1 / test_batch_num:.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}"
    )
    for i in range(len(rate)):
        print(
            f"   relation: {i}\trate: {rate[i].rate:.4f}"
            + f"\trate_f: {rate[i].rate_f:.4f}"
            f"\trate_b: {rate[i].rate_b:.4f}"
            f"\ttotal_f: {rate[i].total_f}"
            f"\ttotal_b: {rate[i].total_b}"
        )

    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5


def pick_step(
    idx_train,
    y_train,
    adj_list,
    size,
    gen_train_flag,
    all_labels,
    b_pick=0.75,
    f_pick=0.45,
):
    max_l = len(y_train)

    degree_train = [1 for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    # lf_train = [0.75 if i == max_l else 0.45 for i in lf_train]
    # lf_train = [0.85 if i == max_l else 0.45 for i in lf_train]
    lf_train = [b_pick if i == max_l else f_pick for i in lf_train]
    if gen_train_flag:
        # lf_train = np.sqrt(lf_train)
        lf_train = lf_train
    # lf_train = np.sqrt((y_train.sum() - len(y_train)) * y_train + len(y_XDtrain))
    smp_prob = np.array(degree_train) / lf_train
    return list((random.choices(idx_train, weights=smp_prob, k=size)))