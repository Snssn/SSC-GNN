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
        """
        将邻接列表转换为稀疏矩阵（scipy.sparse.coo_matrix 格式）
        :param adj_list: 邻接列表，字典格式
        :param num_nodes: 图中的节点数
        :return: 邻接矩阵（稀疏矩阵格式）
        """
        row, col = [], []
        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                row.append(node)
                col.append(neighbor)
        data = np.ones(len(row), dtype=np.float32)  # 设置边权重为1
        adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        return adj_matrix

def load_data(data, prefix="data/"):
    """
    Load graph structure, features, and labels.
    :param data: Dataset name.
    :param prefix: File path.
    :return: graphs, features, and labels.
    """

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
    neg_nodes = cp.deepcopy(nodes)   # 深拷贝，不会改变
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
    """
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    :returns: test or validation results
    """
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
        # turn it into one-hot
        # gnn_pred_one_hot = np.zeros_like(gnn_prob.detach().data.cpu(), dtype=np.int32)
        # gnn_pred_one_hot[np.arange(len(gnn_pred)), gnn_pred] = 1
        # gnn_pred = gnn_pred_one_hot

        # f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
        # acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
        # recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
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
    # auc_label1 = roc_auc_score(labels, np.array(label_list1))
    # ap_label1 = average_precision_score(labels, np.array(label_list1))
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
    # degree_train = [len(adj_list[node]) for node in idx_train]
    # lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    # lf_train2 = (y_train.sum() * y_train) - y_train.sum()
    # lf_train = lf_train + lf_train2
    # lf_train = np.sqrt(lf_train)
    max_l = len(y_train)

    # degree_train = [1 for node in idx_train]
    # lf_train = (y_train.sum()-len(y_train))*y_train + len(y_train)
    # smp_prob = np.array(degree_train) / lf_train
    # return random.choices(idx_train, weights=smp_prob, k=size)

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

    def get_numberof_posandneg(lf0=1, lf1=0.4):
        smp_prob = np.array(degree_train) / (
            [
                lf0 if i == max_l else lf1
                for i in ((y_train.sum() - len(y_train)) * y_train + len(y_train))
            ]
        )
        random.choices(idx_train, weights=smp_prob, k=size)
        ret = (all_labels[random.choices(idx_train, weights=smp_prob, k=size)]).sum()
        print(ret, size - ret)
        print(int(ret), int(size - ret))


# Random state.
RS = 50


def scatter(x, colors):
    # 使用红蓝两种颜色作为调色板
    palette = np.array(['red', 'blue'])  # 直接指定红蓝颜色

    # 确保颜色索引为整数类型
    colors0 = colors.astype(np.int32)

    # 创建散点图
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors0])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("tight")
    ax.axis("off")

    # 添加装饰性矩形（可根据需要保留或删除）
    ax.add_patch(
        plt.Rectangle((100, 100), 200, 100, color="blue", fill=False, linewidth=1)
    )

    # 添加每个类别的标签
    txts = []
    unique_labels = np.unique(colors)  # 获取实际存在的类别标签
    for i in unique_labels:
        # 过滤掉不存在的类别
        if len(x[colors == i]) == 0:
            continue
        # 计算标签位置
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        # 创建文本并设置颜色为对应节点颜色
        txt = ax.text(xtext, ytext, str(i), fontsize=48, color=palette[i])
        # 添加文字效果
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])
        txts.append(txt)

    return f, ax, sc, txts


def getVISUAL(X, y, name, fmt="svg"):
    # digits = load_digits()
    # We first reorder the data points according to the handwritten numbers.
    # temp = digits.data[1]
    # X = np.vstack([digits.data[digits.target == i] for i in range(10)])
    # y = np.hstack([digits.target[digits.target == i]
    #                for i in range(10)])
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    # scatter(digits_proj, y)
    f, ax, sc, txts = scatter(digits_proj, y)

    plt.title(name, y=-0.1, fontsize=40)
    plt.show()
    if fmt == "pdf":
        plt.savefig(name + ".pdf", format="pdf", dpi=1200)
    elif fmt == "svg":
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    else:
        plt.savefig(name + "." + fmt, format=fmt, dpi=1200)


def draw(img, left, right, color):
    # img=cv2.imread(spath)
    img = cv2.rectangle(img, left, right, color, 3)
    return img


import csv

person = [("xxx", 18, 193), ("yyy", 18, 182), ("zzz", 19, 185)]
header = ["name", "age", "height"]


def writeHeader(file_name, header, mode):
    with open(file_name, mode, encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
    file_obj.close()


def writeRow(file_name, row, mode):
    with open(file_name, mode, encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(row)

    file_obj.close()


# !! 修改添加
import random
import dgl
import torch
import numpy as np
import scipy.sparse as sp


def generate_rwr_subgraph(dgl_graph, walk_length=3, num_walks=10):
    nodes = list(range(dgl_graph.number_of_nodes()))  # 全图节点
    rwr_subgraphs = []
    
    for _ in range(num_walks):
        start_node = np.random.choice(nodes)
        trace = [start_node]
        
        for _ in range(walk_length):
            walks = dgl.sampling.random_walk(dgl_graph, [trace[-1]], length=1)

            if len(walks) > 0 and len(walks[0]) > 0:
                next_nodes = walks[0]

                # 确保 next_nodes 是张量，处理时先移到 CPU
                if next_nodes.shape[0] > 1:
                    # 随机选择多个候选中一个
                    next_node = np.random.choice(next_nodes.detach().cpu().numpy())  # 随机选择一个节点
                else:
                    next_node = next_nodes[0].item()  # 如果只有一个元素，安全地调用 item()

                trace.append(next_node)  # 将选择的节点添加到 trace
            else:
                break  # 没有有效的走法，停止

        rwr_subgraphs.append(trace)  # 保存随机游走路径

    return rwr_subgraphs







def aug_random_edge(input_adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    
    # 确保不删除超出边缘数量的数量
    num_drop = int(len(row_idx) * percent)
    num_drop = min(num_drop, len(row_idx))

    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))

    drop_idx = random.sample(edge_index, k=num_drop) if num_drop > 0 else []

    # 删除边
    for idx in drop_idx:
        del edges[idx]

    # 生成新的邻接矩阵
    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0] if new_edges else []
    new_col_idx = new_edges[1] if new_edges else []
    
    data = np.ones(len(new_row_idx)).tolist() if new_row_idx else []
    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape=input_adj.shape)

    # 找出没有连接的边
    row_idx, col_idx = (new_adj.todense() - 1).nonzero()
    no_edges_cells = list(zip(row_idx, col_idx))

    # 确保有足够的节点可供添加边
    if not no_edges_cells:
        print("No available edges to augment.")
        return new_adj

    add_num = num_drop // 2  # 设置添加边数为原计划的一半，确保安全
    add_num = min(add_num, len(no_edges_cells))  # 确保不会超出可添加的边数

    # 选择要添加的边
    add_idx = random.sample(no_edges_cells, add_num) if add_num > 0 else []

    if add_idx:
        new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))
        row_idx = new_row_idx + new_row_idx_1
        col_idx = new_col_idx + new_col_idx_1
        data = np.ones(len(row_idx)).tolist()  # 为新添加的边添加数据
    else:
        row_idx = new_row_idx
        col_idx = new_col_idx

    # 检查长度是否一致
    if len(data) != len(row_idx) or len(data) != len(col_idx):
        print("Length mismatch: data length:", len(data), "row_idx length:", len(row_idx), "col_idx length:", len(col_idx))
        raise ValueError("Row, column, and data arrays must be the same length.")

    new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=input_adj.shape)
    
    return new_adj