#coding: utf-8
import os
import math
import json
import random
random.seed(1234)
import numpy as np
from datetime import datetime
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


def sample_index(ttl_num):
    f1 = int(ttl_num / 10000)
    f2 = int((ttl_num - f1 * 10000) / 100)
    f3 = int((ttl_num - f1 * 10000 - f2 * 100))
    ind1 = random.randint(0, f1)
    if ind1 == f1:
        ind23 = random.randint(0, f2)
        if ind23 == f2:
            ind45 = random.randint(0, f3 - 1)
        else:
            ind45 = random.randint(0, 99)
    else:
        ind23 = random.randint(0, 99)
        ind45 = random.randint(0, 99)
    ind = ind1 * 10000 + ind23 * 100 + ind45
    return ind


class TrainData(Dataset):
    def __init__(self, conf, input_seq, user_seq, item_seq, all_items):
        self.conf = conf
        self.input_seq = input_seq
        self.user_seq = user_seq
        self.item_seq = item_seq
        self.n_all = len(all_items)


    def __len__(self):
        return len(self.input_seq)
    

    def __getitem__(self, idx):
        seq = self.input_seq[idx]
        u = seq[-1]
        i = seq[-2]
        l = seq[-3]
        hist = seq[:-2]
        user_seq = self.user_seq[str(u)]
        item_seq = self.item_seq[str(l)]
  
        k = sample_index(self.n_all)
        while k in user_seq or k in item_seq:
            k = sample_index(self.n_all)

        return u, l, i, int(k)


class TestData(Dataset):
    def __init__(self, conf, input_seq, neg_seq):
        self.conf = conf
        self.input_seq = input_seq
        self.neg_seqs = neg_seq # list of neg samples for each item


    def __len__(self):
        return len(self.input_seq)


    def __getitem__(self, idx):
        seq = self.input_seq[idx]
        u = seq[-1]
        i = seq[-2]
        l = seq[-3]
        hist = seq[:-2]
        ks = [self.neg_seqs[cnt][idx] for cnt in self.neg_seqs.keys()]
        ks = torch.LongTensor(ks)    

        return u, l, i, ks

    
class DGSR_Dataset():
    def __init__(self, conf):
        all_data = self.load_cache_data(conf)
        self.train_seqs, self.val_seqs, self.val_negs, self.test_seqs, self.test_negs, self.user_id_map, self.id_user_map, self.item_id_map, self.id_item_map, self.user_item_set, self.item_item_set = all_data
        self.item_ids = list(self.id_item_map.keys())
        
        self.train_pairs_ui = self.load_cf_data()
        self.train_pairs_ii = self.load_trans_data()

        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)
        self.train_len = len(self.train_seqs)
        
        self.adj_ui, self.adj_iu = self.get_adj(self.train_pairs_ui, self.n_users, self.n_items)
        self.adj_ij, self.adj_ji = self.get_adj(self.train_pairs_ii, self.n_items, self.n_items)

        self.train_set = TrainData(conf, self.train_seqs, self.user_item_set, self.item_item_set, self.item_ids)
        self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10)
        self.test_set = TestData(conf, self.test_seqs, self.test_negs)
        self.test_loader = DataLoader(self.test_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=10)
        self.val_set = TestData(conf, self.val_seqs, self.val_negs)
        self.val_loader = DataLoader(self.val_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=10)
 

    def load_cf_data(self):
        train_pairs = []
        for train_seq in self.train_seqs:
            uid = train_seq[-1]
            iid = train_seq[-2]
            train_pairs.append([uid, iid])
        return np.array(train_pairs)
    
    
    def load_trans_data(self):
        train_pairs = []
        for train_seq in self.train_seqs:
            iid = train_seq[-2]
            lid = train_seq[-3]
            train_pairs.append([lid, iid])
        return np.array(train_pairs)    

    
    def get_adj(self, train_pairs, n_node1, n_node2):
        a_rows = train_pairs[:, 0]
        a_cols = train_pairs[:, 1]
        a_vals = [1.] * len(a_rows)

        b_rows = a_cols
        b_cols = a_rows
        b_vals = [1.] * len(b_rows)

        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_node1, n_node2))
        b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_node2, n_node1))

        a_adj = self.get_lap(a_adj)
        b_adj = self.get_lap(b_adj)

        return a_adj, b_adj
    
    
    def get_lap(self, adj):
        def bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum + 0.00000001, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

            return bi_lap.tocoo()
        
        def si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum + 0.00000001, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)

            return norm_adj.tocoo()
        
        lap = si_norm_lap(adj)
            
        return lap


    def load_cache_data(self, conf):
        target_path = conf["target_path"]
        val_negs, test_negs = {}, {}
        train_seqs = json.load(open(target_path + "train_seqs_%d_large.json"%conf['seq_len']))
        if conf["dataset"] in ["ifashion"]: # val_negs and test_negs for ifashion is a dict with more than one set of negs, we take the first set.
            [val_seqs, val_negs[0]] = json.load(open(target_path + "val_seqs_%d.json"%conf['seq_len']))
            [test_seqs, test_negs[0]] = json.load(open(target_path + "test_seqs_%d.json"%conf['seq_len']))
        else: # val_negs and test_negs for amazon is a list, which is the only neg set
            val_seqs = json.load(open(target_path + "val_seqs_%d.json"%conf['seq_len']))
            val_negs = json.load(open(target_path + "val_negs_%d.json"%conf['seq_len']))
            test_seqs= json.load(open(target_path + "test_seqs_%d.json"%conf['seq_len']))
            test_negs = json.load(open(target_path + "test_negs_%d.json"%conf['seq_len']))
        user_id_map = json.load(open(target_path + "user_id_map_%d.json"%conf['seq_len']))
        id_user_map = json.load(open(target_path + "id_user_map_%d.json"%conf['seq_len']))
        item_id_map = json.load(open(target_path + "item_id_map_%d.json"%conf['seq_len']))
        id_item_map = json.load(open(target_path + "id_item_map_%d.json"%conf['seq_len']))
        user_item_set = json.load(open(target_path + "user_item_set_%d.json"%conf['seq_len']))    
        item_item_set = json.load(open(target_path + "item_pos_item_set_%d.json"%conf['seq_len']))
    
        # convert values into sets to accelerrate the lookup
        user_item_set = {user: set(items) for user, items in user_item_set.items()}
        item_item_set = {item: set(items) for item, items in item_item_set.items()}
    
        return train_seqs, val_seqs, val_negs, test_seqs, test_negs, user_id_map, id_user_map, item_id_map, id_item_map, user_item_set, item_item_set
