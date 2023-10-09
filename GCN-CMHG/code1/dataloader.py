import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

import pandas as pd


#torch.set_default_tensor_type(torch.DoubleTensor)


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        self.Graph_FA = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")

        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users))) #allPosItems
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self,use_PA):
        '''
        功能：
        根据用户-物品交互信息，构造不含节点自连接的用户-物品交互异构图、含节点自连接的用户-物品交互异构图以及异构全相邻图/异构部分相邻图
        输出：
        self.Graph：不含节点自连接的用户-物品交互异构图
        self.Graph_FA：当使用HFAA时为异构全相邻图，当使用HPAA时为异构部分相邻图
        self.Graph_WF: 含节点自连接的用户-物品交互异构图
        todo:
            将用户-物品交互异构图的含与不含节点自连接的变化放到model.py完成，以降低空间复杂度
        '''
        print('--------------------------------------lastfm--------------------------------------------')
        if self.Graph is None:
            '''__构造不含节点自连接的用户-物品交互异构图__'''
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])

            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            
            dense = self.Graph.to_dense()
            dense1 = dense
            
            D = torch.sum(dense, dim=1).float()
            
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()

            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
            
            '''__GraphWF__构造含节点自连接的用户-物品交互异构图__'''
            dense2 = dense + torch.eye(self.n_users+self.m_items)
            index = dense2.nonzero()
            data  = dense2[dense2 >= 1e-9]
            assert len(index) == len(data)
            self.Graph_WF = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph_WF = self.Graph_WF.coalesce().to(world.device)
            
            '''__graphFA__构造异构全/部分相邻图__'''
            dense1[:self.n_users,:self.n_users] = torch.ones(self.n_users, self.n_users)
            dense1[self.n_users:,self.n_users:] = torch.ones(self.m_items, self.m_items)
            dense1[:self.n_users,self.n_users:] = torch.zeros(self.n_users, self.m_items)
            dense1[self.n_users:,:self.n_users] = torch.zeros(self.m_items, self.n_users)
            
            #use_PA = False#regional central node
            print('use_PA',use_PA)
            if use_PA == 'True':
                
                Graph_tmp = pd.read_csv("regional_central_node_lastfm.csv")
                Graph_tmp_arr = pd.to_numeric(Graph_tmp["0"])
                Graph_tmp = Graph_tmp_arr.tolist()
                print(self.Graph.shape[0])
                Graph_ = []
                for i in range(self.Graph.shape[0]):
                    Graph_.append(Graph_tmp)
                print(len(Graph_),len(Graph_[0]),sum(Graph_[10]))
                Graph_tensor = torch.tensor(Graph_)
                #Graph_tensor = Graph_tensor.to(world.device)
                Graph_tensor[:1892,1892:] = torch.zeros(1892, 4489)
                Graph_tensor[1892:,:1892] = torch.zeros(4489,1892)
                #Graph_tensor[:1892,:1892] = torch.ones(1892, 1892)
                #Graph_tensor[1892:,1892:] = torch.ones(4489,4489)
                Graph_tensor = Graph_tensor+torch.eye(self.n_users+self.m_items)
                print(Graph_tensor)
                Graph_tensor[Graph_tensor==2.] = 1.
                print(Graph_tensor)
                dense1 = Graph_tensor
                #dense1 = dense1 + torch.eye(self.n_users+self.m_items)
                print('The heterogeneous partial adjacent graph is successfully constructed...')
            D = torch.sum(dense1, dim=1).float()
            print('dense1',dense1)
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense1/D_sqrt
            dense = dense/D_sqrt.t()
            dense = dense + torch.eye(self.n_users+self.m_items) - (torch.eye(self.n_users+self.m_items))*0.0005
            
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph_FA = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph_FA = self.Graph_FA.coalesce().to(world.device)
            
            
        print('self.Graph_FA',type(self.Graph_FA),self.Graph_FA.shape)
        
        print(self.Graph_FA)
        print('self.Graph',self.Graph)
        #self.Graph_FA = Graph_tensor
        return self.Graph, self.Graph_FA,self.Graph_WF

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])#user对应的positems的索引
        return posItems#(users*items)
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        self.Graph_PA = None
        self.Graph_WF = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        use_PA = 'True'
        use_WF = 'True'
        #print('111111',len(self.trainUser))#810128
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat0426.npz')
                #pre_adj_mat_FA = sp.load_npz(self.path + '/s_pre_adj_mat_fa08081.npz')
                print("successfully loaded.....")
                print(use_PA == 'True')
                norm_adj = pre_adj_mat
                if use_PA == 'True':
                    pre_adj_mat_PA = sp.load_npz(self.path + '/s_pre_adj_mat_HPAA.npz')
                    norm_adj_PA = pre_adj_mat_PA
                    print("The heterogeneous partial adjacent graph is successfully loaded...")
                if use_WF == 'True':
                    pre_adj_mat_WF = sp.load_npz(self.path + '/s_pre_adj_mat_WF.npz')
                    norm_adj_WF = pre_adj_mat_WF
                    print("The graph with self-connection is successfully loaded...")
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)#用dok_matrix表示的邻接矩阵
                adj_mat = adj_mat.tolil()

                R = self.UserItemNet.tolil()

                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                #adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)

                norm_adj = norm_adj.dot(d_mat)

                norm_adj = norm_adj.tocsr()
                
                

                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat0426.npz', norm_adj)
                
                if use_WF == 'True':
                    adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                    rowsum_WF = np.array(adj_mat.sum(axis=1))
                    d_inv_WF = np.power(rowsum_WF, -0.5).flatten()
                    d_inv_WF[np.isinf(d_inv_WF)] = 0.
                    d_mat_WF = sp.diags(d_inv_WF)
                    
                    norm_adj_WF = d_mat_WF.dot(adj_mat)
    
                    norm_adj_WF = norm_adj_WF.dot(d_mat_WF)
    
                    norm_adj_WF = norm_adj_WF.tocsr()
                    
                    sp.save_npz(self.path + '/s_pre_adj_mat_WF.npz', norm_adj_WF)
                    
            print('self.split',self.split)
            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                print(norm_adj)
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix", self.Graph)
                if use_PA == 'True':
                    print(norm_adj_PA)
                    self.Graph_PA = self._convert_sp_mat_to_sp_tensor(norm_adj_PA)
                    self.Graph_PA = self.Graph_PA.coalesce().to(world.device)
                    print("The heterogeneous partial adjacent graph is ready", self.Graph_PA)
                if use_WF == 'True':
                    self.Graph_WF = self._convert_sp_mat_to_sp_tensor(norm_adj_WF)
                    self.Graph_WF = self.Graph_WF.coalesce().to(world.device)
                    print("The graph that retain self-retain is ready", self.Graph_WF)
        #print('self.Graph.shape',self.Graph.shape,self.Graph_PA.shape)

        return self.Graph, self.Graph_PA, self.Graph_WF#, adj_mat

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
