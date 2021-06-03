import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
import argparse
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')

from models.DGSR import DGSR
from utility import DGSR_Dataset

torch.manual_seed(0)


def get_cmd(): 
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default=0, type=str, help="which gpu to use")  
    parser.add_argument("-d", "--dataset", default="ifashion", type=str, help="dataset, options: ifashion, amazon")
    parser.add_argument("-l", "--seq_len", default=5, type=int, help="input seq len, options: 5, 8")
    parser.add_argument("-e", "--emb_size", default=50, type=int, help="the embedding size")
    args = parser.parse_args()

    return args


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["mrr"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key in best_metrics:
            for metric in best_metrics[key]:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform
     
    
def mk_save_dir(types, root_path, settings):
    output_dir = "./%s/%s"%(types, root_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "%s"%("__".join([str(i) for i in settings]))

    return output_file


def write_log(log, topk, step, val_metrics, test_metrics):
    def write_log_metrics(metric):
        log.add_scalar("%s_%s/Val" %(metric, topk), val_metrics[metric][topk], step)
        log.add_scalar("%s_%s/Test" %(metric, topk), test_metrics[metric][topk], step)
    write_log_metrics("mrr")
    write_log_metrics("recall")
    write_log_metrics("ndcg")


def train(conf):
    dataset = DGSR_Dataset(conf)
    conf['n_users'] = dataset.n_users
    conf['n_items'] = dataset.n_items
    conf["num_train_seqs"] = len(dataset.train_seqs)
    conf["num_val_seqs"] = len(dataset.val_seqs)
    conf["num_test_seqs"] = len(dataset.test_seqs)
 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(conf["gpu"])
    conf["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for k, v in conf.items():
        print(k, v)

    root_path = "%s/seq_len_%d/dgsr/" %(conf['dataset'].upper(), conf["seq_len"])
    settings = ["EmbSize%d" %(conf["emb_size"])]
    settings.append("UIL%d" %(conf["ui_layers"]))
    settings.append("IIL%d" %(conf["ii_layers"]))
    settings.append("LR%.4f" %(conf["lr"]))

    model = DGSR(conf, dataset.adj_ui, dataset.adj_iu, dataset.adj_ij, dataset.adj_ji)
    model.to(device=conf["device"])

    performance_file = mk_save_dir("performance", root_path, settings)
    result_file = mk_save_dir("results", root_path, settings)
    log_file = mk_save_dir("logs", root_path, settings)
    model_save_file = mk_save_dir("model_saves", root_path, settings)
    log = SummaryWriter(log_file)    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
    print("%s start training ... "%datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = 0
    
    for epoch in range(conf["num_epoches"]):
        model.train(True)
        loss_print = 0
        for batch_cnt, batch in enumerate(dataset.train_loader):
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(conf["device"]) for x in batch]
            loss = model(batch)
            loss_scalar = loss.detach().cpu()
            loss_print += loss_scalar
            loss.backward()
            optimizer.step()
            log.add_scalar("Loss/Loss", loss_scalar, batch_cnt+epoch*len(dataset.train_loader))

        output_f = open(performance_file, "a")
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = '%s  epoch %d, %s/len_%d/dgsr/%s, loss %.4f' %(curr_time, epoch, conf["dataset"], conf["seq_len"], "_".join([str(i) for i in settings]), loss_print/(batch_cnt+1))
        output_f.write(log_str + "\n")
        print(log_str)
        output_f.close()
        if (epoch + 1) % conf["test_interval"] == 0:
            model.eval()
            best_metrics, best_perform, best_epoch = evaluate(model, dataset, conf, log, performance_file, result_file, model_save_file, epoch, batch_cnt, best_metrics, best_perform, best_epoch)

            
def evaluate(model, dataset, conf, log, performance_file, result_file, model_save_file, epoch, batch_cnt, best_metrics, best_perform, best_epoch): 
    metrics = {}
    metrics["val"], _, _ = rank(model, dataset.val_loader, conf, which="val")
    metrics["test"], all_grd, all_pred = rank(model, dataset.test_loader, conf, which="test")
    step = batch_cnt + epoch*dataset.train_len/conf["batch_size"]
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for topk in conf['topk']:
        print("%s, Val  res: rec_%d: %f, mrr_%d: %f, ndcg_%d: %f" %(curr_time, topk, metrics["val"]["recall"][topk], topk, metrics["val"]["mrr"][topk], topk, metrics["val"]["ndcg"][topk]))
        print("%s, Test res: rec_%d: %f, mrr_%d: %f, ndcg_%d: %f" %(curr_time, topk, metrics["test"]["recall"][topk], topk, metrics["test"]["mrr"][topk], topk, metrics["test"]["ndcg"][topk]))
        write_log(log, topk, step, metrics["val"], metrics["test"])

    topk_ = conf['topk'][1]
    print("top%d as the final evaluation standard" %(topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_] and metrics["val"]["mrr"][topk_] > best_metrics["val"]["mrr"][topk_]:
        best_epoch = epoch
        output_f = open(performance_file, "a")
        for topk in conf['topk']:
            for key in best_metrics:
                for metric in best_metrics[key]:                    
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]
            
            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, MRR_T=%.5f, NDCG_T=%.5f"%(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["mrr"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, MRR_V=%.5f, NDCG_V=%.5f"%(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["mrr"][topk], best_metrics["val"]["ndcg"][topk])
            output_f.write(best_perform["val"][topk] + "\n")
            output_f.write(best_perform["test"][topk] + "\n")

        output_f.write("\n")
        output_f.close()
        torch.save(model.state_dict(), model_save_file+"model.state_dict")

        np.save(result_file + "all_grd.npy", all_grd)
        for i in all_pred:
            np.save(result_file + "all_pred_%d"%i, all_pred[i])

    print(best_perform["val"][topk_])
    print(best_perform["test"][topk_])

    return best_metrics, best_perform, best_epoch


def rank(model, test_loader, conf, which="test"):
    trg_item_pred = {}
    trg_item_cnt = []
    trg_item_grd = []
    user = []
    pos_item = []
    neg_items = []

    ui_embedding_i = model.ui_embedding_i
    ii_embedding_l = model.ii_embedding_l
    ii_embedding_i = model.ii_embedding_i

    ui_rep_u = model.mp_ui(model.ui_embedding_u, ui_embedding_i)
    ui_rep_i = model.mp_iu(ui_embedding_i, model.ui_embedding_u)
    ii_rep_l = model.mp_ij(ii_embedding_l, ii_embedding_i)
    ii_rep_i = model.mp_ji(ii_embedding_i, ii_embedding_l)
    
    for batch_cnt, batch in enumerate(test_loader):
        batch = [i.to(conf['device']) for i in batch]
        u, l, i, ks = batch
        [bs, neg_set_num, neg_num] = ks.size()
        
        u_rep, pos_iu_rep = None, None
        u_rep = ui_rep_u[u]
        pos_iu_rep = ui_rep_i[i]
        l_rep, pos_il_rep = None, None
        l_rep = ii_rep_l[l]
        pos_il_rep = ii_rep_i[i]
        pos_score = model.predict(u_rep, pos_iu_rep, l_rep, pos_il_rep).unsqueeze(-1).detach()

        u_rep = u_rep.unsqueeze(1).expand(-1, neg_num, -1)
        l_rep = l_rep.unsqueeze(1).expand(-1, neg_num, -1)
        for neg_set_cnt in range(neg_set_num):
            if neg_set_cnt not in trg_item_pred:
                trg_item_pred[neg_set_cnt] = []
            k = ks[:, neg_set_cnt, :]
            neg_iu_rep, neg_il_rep = None, None
            neg_iu_rep = ui_rep_i[k]
            neg_il_rep = ii_rep_i[k]
            neg_score = model.predict(u_rep, neg_iu_rep, l_rep, neg_il_rep).detach()
            score = torch.cat([pos_score, neg_score], dim=-1)
            _, tops = torch.topk(score, k=max(conf["topk"]), dim=-1)
            tops = tops.cpu().numpy()
            trg_item_pred[neg_set_cnt].append(tops)
            
        trg_item_grd.append([0] * np.shape(tops)[0])
        trg_item_cnt.append([1] * np.shape(tops)[0])

    grd = np.concatenate(trg_item_grd, axis=0)
    grd_cnt = np.concatenate(trg_item_cnt, axis=0)
    pred = {}
    for neg_set_cnt in range(neg_set_num):
        pred[neg_set_cnt] = np.concatenate(trg_item_pred[neg_set_cnt], axis=0)
    
    REC, MRR, NDCG = get_metrics(grd, grd_cnt, pred, neg_set_num, conf["topk"])    
    metrics = {}
    metrics["recall"] = REC
    metrics["mrr"] = MRR
    metrics["ndcg"] = NDCG

    return metrics, grd, pred


def get_metrics(grd, grd_cnt, pred, neg_set_num, topks):
    REC, MRR, NDCG = {}, {}, {}
    for topk in topks:
        REC[topk] = []
        MRR[topk] = []
        NDCG[topk] = []
        for neg_cnt in range(neg_set_num):
            rec_, mrr_, ndcg_ = [], [], []
            for each_grd, each_grd_cnt, each_pred in zip(grd, grd_cnt, pred[neg_cnt]):
                ndcg_.append(getNDCG(each_pred[:topk], [each_grd][:each_grd_cnt]))
                hit, mrr = getHIT_MRR(each_pred[:topk], [each_grd][:each_grd_cnt])
                rec_.append(hit)
                mrr_.append(mrr)

            mrr_ = np.mean(mrr_)
            rec_ = np.mean(rec_)
            ndcg_ = np.mean(ndcg_)
            REC[topk].append(rec_)
            MRR[topk].append(mrr_)
            NDCG[topk].append(ndcg_)
        REC[topk] = np.mean(REC[topk])
        MRR[topk] = np.mean(MRR[topk])
        NDCG[topk] = np.mean(NDCG[topk])

    return REC, MRR, NDCG


def getHIT_MRR(pred, target_items):
    hit= 0.
    mrr = 0.
    p_1 = []
    for p in range(len(pred)):
        pre = pred[p]
        if pre in target_items:
            hit += 1
            if pre not in p_1:
                p_1.append(pre)
                mrr = 1./(p+1)

    return hit, mrr


def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if item_id not in target_items:
            continue
        rank = i + 1
        dcg += 1./math.log(rank+1, 2)

    return dcg/idcg


def IDCG(n):
    idcg = 0.
    for i in range(n):
        idcg += 1./math.log(i+2, 2)

    return idcg

                    
def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("Load config.yaml done!")
    paras = get_cmd().__dict__

    for k in paras:
        conf[k] = paras[k]
    dataset = conf["dataset"]    
    # below are the best settings according to ablation studies
    if dataset == "ifashion":
        conf["ii_layers"] = 3
        conf["ui_layers"] = 1
    elif dataset == "amazon":
        conf["ii_layers"] = 2
        conf["ui_layers"] = 1

    for k, v in conf[dataset].items():
        conf[k] = v
    for seq_len in [5, 8]:
        conf["seq_len"] = seq_len
        train(conf)


if __name__ == "__main__":
    main()        
