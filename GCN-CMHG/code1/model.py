import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


#torch.set_default_tensor_type(torch.DoubleTensor)

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):

        users = users.long()
        users_emb = self.embedding_user(users)

        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        #print('opopop',items)
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        #self.Graph = self.dataset.getSparseGraph()
        
        self.use_FA = self.config['use_HFAA']
        self.use_PA = self.config['use_HPAA']
        self.use_WF = self.config['use_PLNS']
        self.use_side = self.config['ues_side']
        self.AAlayer = self.config['AAlayer']
        self.PLNSlayer = self.config['PLNSlayer']
        
        if world.dataset == 'lastfm':
            self.Graph,self.Graph_FA, self.Graph_WF = self.dataset.getSparseGraph(self.use_PA)
            self.Graph_side = None
        elif world.dataset == 'gowalla'or world.dataset == 'ml1m_dense' or world.dataset == 'Movies_and_TV' or world.dataset == 'yelp2018':
            self.Graph,self.Graph_FA, self.Graph_WF = self.dataset.getSparseGraph()
            self.Graph_side = None, None
            print('self.use_FA',self.use_FA)
        else:
            self.Graph,self.Graph_FA, self.Graph_WF, self.Graph_side = self.dataset.getSparseGraph(self.use_PA)
        #self.Grapg_feature = self.dataset.getSparseGraph1()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        print('__dropout_x')
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        print("__dropout")
        if self.A_split:
            graph = []
            for g in self.Graph:
                print('g',g)
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight#users_emb:(29858,64)
        items_emb = self.embedding_item.weight#items_emb:(40981,64)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        '''side'''
        if self.use_side == 'True':
            Graph_side = self.Graph_side
            all_emb = torch.sparse.mm(Graph_side.to(torch.float32), all_emb)
            #embs.append(all_emb1)
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:            
            g_droped = self.Graph  
            Graph_FA = self.Graph_FA
            Graph_WF = self.Graph_WF
        for layer in range(self.n_layers):#遍历每一层
            if layer >= self.n_layers-self.PLNSlayer and self.use_WF == 'True':#if layer >= 2 and self.use_WF == True:
                all_emb = torch.sparse.mm(Graph_WF.to(torch.float32), all_emb)
                embs.append(all_emb)
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)#g_droped:(70839,70839), all_emb:(70839,64)
                embs.append(all_emb)
            if layer >= self.n_layers-self.AAlayer and self.use_FA == 'True':
                #print('HPAA')
                all_emb = torch.sparse.mm(Graph_FA.to(torch.float32), all_emb)
                embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def NGCF(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        '''side'''
        Graph_side = self.Graph_side
        if self.use_side == True:
            all_emb = torch.sparse.mm(Graph_side.to(torch.float32), all_emb)
            #embs.append(all_emb1)
        if self.config['dropout']:
            
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:            
            g_droped = self.Graph  
            Graph_FA = self.Graph_FA
            Graph_WF = self.Graph_WF
        for layer in range(self.n_layers):
            if layer >= 2 and self.use_WF == True:
                all_emb = torch.sparse.mm(Graph_WF.to(torch.float32), all_emb)
                embs.append(all_emb)
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            if layer >= self.n_layers-1 and self.use_FA == True:
                all_emb = torch.sparse.mm(Graph_FA.to(torch.float32), all_emb)
                embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        #print(" getEmbedding")
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        #print("bpr_loss")
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
