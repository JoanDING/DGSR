import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def bpr_loss(pos_score, neg_score):
    loss = - F.logsigmoid(pos_score - neg_score)
    loss = torch.mean(loss)

    return loss


def convert_sp_mat_to_sp_tensor(X, shape=-1):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy((coo.data)).float()
    if shape == -1:
        res = torch.sparse.FloatTensor(i, v, coo.shape)
    else:
        res = torch.sparse.FloatTensor(i, v, shape)
    res = Parameter(res, requires_grad=False)

    return res


class LightGCN(nn.Module):
    def __init__(self, A, layers):
        super(LightGCN, self).__init__()
        self.A = A
        self.layers = layers

    
    def forward(self, ego_embeddings, nei_embeddings):    
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            side_embeddings = torch.spmm(self.A, nei_embeddings)
            nei_embeddings = side_embeddings
            norm_embeddings = F.normalize(nei_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings] # this is new compared to the lightgcn code, otherwise the loss is too big 
        all_embeddings = torch.stack(all_embeddings, 1)    
        all_embeddings = torch.mean(all_embeddings, dim=1)

        return all_embeddings


class LightGCN_bi(nn.Module):
    def __init__(self, A, A_t, layers):
        super(LightGCN_bi, self).__init__()
        self.A = A
        self.A_t = A_t
        self.layers = layers
        assert layers <= 3, "only support <=3 layers"
    
    def forward(self, ego_embeddings, nei_embeddings):
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            if k == 0:
                side_embeddings = torch.spmm(self.A, nei_embeddings)
                norm_embeddings = F.normalize(side_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]
            if k == 1:
                side_embeddings = torch.spmm(self.A_t, ego_embeddings)
                side_embeddings = F.normalize(side_embeddings, p=2, dim=1)
                side_embeddings = torch.spmm(self.A, side_embeddings)
                norm_embeddings = F.normalize(side_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings] 
            if k == 2:
                side_embeddings = torch.spmm(self.A, nei_embeddings)
                side_embeddings = F.normalize(side_embeddings, p=2, dim=1)
                side_embeddings = torch.spmm(self.A_t, side_embeddings)
                side_embeddings = F.normalize(side_embeddings, p=2, dim=1)
                side_embeddings = torch.spmm(self.A, side_embeddings)
                norm_embeddings = F.normalize(side_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]
        all_embeddings = torch.stack(all_embeddings, 1)    
        all_embeddings = torch.mean(all_embeddings, dim=1)

        return all_embeddings


class DGSR(nn.Module):
    def __init__(self, conf, adj_ui, adj_iu, adj_ij, adj_ji):
        super(DGSR, self).__init__()
        self.conf = conf      
        self.n_users = conf["n_users"]
        self.n_items = conf["n_items"]
        self.emb_size = conf["emb_size"]
        self.ui_layers = conf["ui_layers"]
        self.ii_layers = conf["ii_layers"]
        self.device = conf["device"]

        self.adj_ui = convert_sp_mat_to_sp_tensor(adj_ui, [self.n_users, self.n_items]).to(self.device)
        self.adj_iu = convert_sp_mat_to_sp_tensor(adj_iu, [self.n_items, self.n_users]).to(self.device)
        self.adj_ij = convert_sp_mat_to_sp_tensor(adj_ij, [self.n_items, self.n_items]).to(self.device)
        self.adj_ji = convert_sp_mat_to_sp_tensor(adj_ji, [self.n_items, self.n_items]).to(self.device)
        self.mp_ui = LightGCN_bi(self.adj_ui, self.adj_iu, self.ui_layers)
        self.mp_iu = LightGCN_bi(self.adj_iu, self.adj_ui, self.ui_layers)
        self.mp_ij = LightGCN(self.adj_ij, self.ii_layers)
        self.mp_ji = LightGCN(self.adj_ji, self.ii_layers)
        self.ui_embedding_u, self.ui_embedding_i, self.ii_embedding_l, self.ii_embedding_i = self.init_embeddings()


    def init_embeddings(self):
        initializer = nn.init.xavier_uniform_
        ui_embedding_u, ui_embedding_i = None, None
        ui_embedding_u = initializer(torch.empty(self.n_users, self.emb_size))
        ui_embedding_u = Parameter(ui_embedding_u, requires_grad=True)
        ui_embedding_i = initializer(torch.empty(self.n_items, self.emb_size))
        ui_embedding_i = Parameter(ui_embedding_i, requires_grad=True)
        ii_embedding_l, ii_embedding_i = None, None
        ii_embedding_l = initializer(torch.empty(self.n_items, self.emb_size))
        ii_embedding_l = Parameter(ii_embedding_l, requires_grad=True)
        ii_embedding_i = initializer(torch.empty(self.n_items, self.emb_size))   
        ii_embedding_i = Parameter(ii_embedding_i, requires_grad=True)

        return ui_embedding_u, ui_embedding_i, ii_embedding_l, ii_embedding_i
        
            
    def predict(self, u_rep, iu_rep, l_rep, il_rep):
        ui = u_rep * iu_rep
        ii = l_rep * il_rep
        pred = ui + ii
        pred = torch.sum(pred, dim=-1)

        return pred
    

    def get_rep(self, batch):
        u, l, pos_i, neg_i = batch
        ui_embedding_i = self.ui_embedding_i
        ui_rep_u = self.mp_ui(self.ui_embedding_u, ui_embedding_i)
        ui_rep_i = self.mp_iu(ui_embedding_i, self.ui_embedding_u)
        u_rep = ui_rep_u[u]
        pos_iu_rep = ui_rep_i[pos_i]
        neg_iu_rep = ui_rep_i[neg_i]
        ii_embedding_l = self.ii_embedding_l
        ii_embedding_i = self.ii_embedding_i
        ii_rep_l = self.mp_ij(ii_embedding_l, ii_embedding_i)
        ii_rep_i = self.mp_ji(ii_embedding_i, ii_embedding_l)
        l_rep = ii_rep_l[l]
        pos_il_rep = ii_rep_i[pos_i]
        neg_il_rep = ii_rep_i[neg_i]
            
        return u_rep, pos_iu_rep, neg_iu_rep, l_rep, pos_il_rep, neg_il_rep

    
    def forward(self, batch):
        u_rep, pos_iu_rep, neg_iu_rep, l_rep, pos_il_rep, neg_il_rep = self.get_rep(batch)
        pos_pred = self.predict(u_rep, pos_iu_rep, l_rep, pos_il_rep)
        neg_pred = self.predict(u_rep, neg_iu_rep, l_rep, neg_il_rep)
        loss = bpr_loss(pos_pred, neg_pred)

        return loss
