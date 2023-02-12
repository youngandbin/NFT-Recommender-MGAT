import math
# from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from GraphGAT import GraphGAT  

class MMMO(torch.nn.Module):
    def __init__(self, features, user_features, edge_index, batch_size, num_user, num_item, reg_parm, dim_x, DROPOUT, path=None, cluster_dict=None):
        super(MMMO, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.reg_parm = reg_parm

        self.edge_index = edge_index[:,:2]
        self.edge_index = torch.tensor(self.edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        v_feat, t_feat, p_feat, tr_feat = features
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
        self.p_feat = torch.tensor(p_feat, dtype=torch.float).cuda()
        self.tr_feat = torch.tensor(tr_feat, dtype=torch.float).cuda()

        self.user_features = torch.tensor(user_features, dtype=torch.float).cuda()

        self.v_gnn = GAT(self.v_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, dim_latent=1024)
        self.t_gnn = GAT(self.t_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, dim_latent=1500)
        self.p_gnn = GAT(self.p_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, dim_latent=64)
        self.tr_gnn = GAT(self.tr_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, dim_latent=64)

        self.id_embedding = nn.Embedding(num_user+num_item, dim_x)
        # print('self.id_embedding: ', self.id_embedding)
        nn.init.xavier_normal_(self.id_embedding.weight)

        #self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()

        # read dataset/indices.npy
        # self.indices = np.load(path+'indices.npy', allow_pickle=True)

        # linear layer
        self.MLP_price = MLP_price(dim_in=dim_x*2, dim_out=1)

        # cluster dict
        self.cluster_dict = cluster_dict

        # linear layer (for attention)
        self.q_fc = nn.Linear(dim_x, dim_x, bias=False) # (d_model, d_model)
        self.k_fc = nn.Linear(dim_x, dim_x, bias=False) # (d_model, d_model)
        self.v_fc = nn.Linear(dim_x, dim_x, bias=False) # (d_model, d_model)


    def forward(self, user_nodes, pos_items, neg_items):

        # print('user_nodes: ', user_nodes.shape) # torch.Size([2048])
        # print('pos_items: ', pos_items.shape) # torch.Size([2048])
        # print('neg_items: ', neg_items.shape) # torch.Size([2048])        
        v_rep = self.v_gnn(self.id_embedding)
        t_rep = self.t_gnn(self.id_embedding) 
        p_rep = self.p_gnn(self.id_embedding)
        tr_rep = self.tr_gnn(self.id_embedding)
        self.v_representation = v_rep
        self.t_representation = t_rep
        self.p_representation = p_rep
        self.tr_representation = tr_rep
        # print('tr_rep: ', tr_rep.shape) # torch.Size([num_user+num_item, 512])

        representation = (v_rep + t_rep + p_rep + tr_rep) / 4 #torch.max_pool2d((v_rep, a_rep, t_rep))#max()#torch.cat((v_rep, a_rep, t_rep), dim=1)
        # print('representation: ', representation.shape) # torch.Size([num_user+num_item, 512])
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]

        # QUERY
        Q = user_tensor
        # print('Q: ', Q.shape) # torch.Size([2048, 512])
        # KEY, VALUE
        pos_tensor_v = v_rep[pos_items]
        pos_tensor_t = t_rep[pos_items]
        pos_tensor_p = p_rep[pos_items]
        pos_tensor_tr = tr_rep[pos_items]
        # print('pos_tensor_tr: ', pos_tensor_tr.shape) # torch.Size([2048, 512])

        # create a matrix where each row is the average of pos_tensor_v, pos_tensor_t, pos_tensor_p, pos_tensor_tr
        K = torch.stack([pos_tensor_v.mean(dim=0), pos_tensor_t.mean(dim=0), pos_tensor_p.mean(dim=0), pos_tensor_tr.mean(dim=0)], dim=0)
        # copy K
        V = K.clone()
        # print('K: ', K.shape) # torch.Size([4, 512])

        # transform
        Q = self.q_fc(Q)
        # print('Q transformed: ', Q.shape) # torch.Size([2048, 512])
        K = self.k_fc(K)
        # print('K transformed: ', K.shape) # torch.Size([4, 512])
        V = self.v_fc(V)

        # calculate attention
        attention = torch.matmul(Q, K.transpose(1, 0)) / np.sqrt(K.shape[1])
        # print('attention: ', attention.shape) # torch.Size([2048, 4])
        attention = F.softmax(attention, dim=1)
        attention = torch.matmul(attention, V)
        # print('attention: ', attention.shape) # torch.Size([2048, 512])
        user_tensor = attention

        

        # 1) BPR loss
        pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
        neg_score = torch.sum(user_tensor * neg_tensor, dim=1)
        
        # 2) Price loss
        user_pos_tensor = torch.cat((user_tensor, pos_tensor), dim=1)
        # linear layer 
        pred_price = self.MLP_price(user_pos_tensor)
        
        return pos_scores, neg_score, representation, pred_price

    def loss(self, data):
        user, pos_items, neg_items, labels = data
        pos_scores, neg_scores,representation, pred_price = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda())

        # 1) BPR loss
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg = (torch.norm(representation[user])**2
        +torch.norm(representation[pos_items])**2
        +torch.norm(representation[neg_items])**2) / 2
        loss_reg = self.reg_parm*reg/self.batch_size

        loss_BPR = loss_value + loss_reg

        # 2) BCE loss
        loss_Price = F.binary_cross_entropy_with_logits(pred_price.float(), labels.unsqueeze(1).float().cuda())

        return loss_BPR, loss_Price


    def accuracy(self, dataset, indices, topk=10, neg_num=100):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        # bar = tqdm(total=len(dataset))
        recall_list = np.array([0.0, 0.0, 0.0, 0.0])
        ndcg_list = np.array([0.0, 0.0, 0.0, 0.0])


        for data in dataset:
            # bar.update(1)

            sum_item += 1
            user = data[0]
            pos_items = data[1:]
      
            # minus from self.indices to pos_items
            neg_items = [x for x in indices if x not in pos_items]
            neg_items = neg_items[:neg_num]
            neg_items = list(neg_items)

            batch_user_tensor = torch.tensor(user).cuda() 
            batch_pos_tensor = torch.tensor(pos_items).cuda()
            batch_neg_tensor = torch.tensor(neg_items).cuda()

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

             # make the topk items using the price score

            pos_price = self.MLP_price(torch.cat((user_embed.repeat(len(pos_items),1), pos_v_embed), dim=1))
            neg_price = self.MLP_price(torch.cat((user_embed.repeat(neg_num,1), neg_v_embed), dim=1))


            # make the topk items using the rank score
            # save the result from topk 5 to topk 10
            for k, topk in enumerate(range(5, 21, 5)):
                _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
                index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
                
                num_hit = len(index_set.difference(all_set))
                # sum_pre += float(num_hit/topk)
      
                recall = float(num_hit/num_pos)
                ndcg_score = 0.0
                for i in range(num_pos):
                    label_pos = neg_num + i
                    if label_pos in index_of_rank_list:
                        index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                        ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
                ndcg = ndcg_score/num_pos

                # append to the list
                recall_list[k] += recall
                ndcg_list[k] += ndcg

        return recall_list/sum_item, ndcg_list/sum_item

    def attention_score(self, dataset):
        attention_score = {}
        for data in dataset:
            # bar.update(1)
            user = data[0]
            pos_items = data[1]

            v_rep = self.v_representation
            t_rep = self.t_representation
            p_rep = self.p_representation 
            tr_rep = self.tr_representation 

            Q = self.result_embed[user]

            pos_tensor_v = v_rep[pos_items]
            pos_tensor_t = t_rep[pos_items]
            pos_tensor_p = p_rep[pos_items]
            pos_tensor_tr = tr_rep[pos_items]

            K = torch.stack([pos_tensor_v, pos_tensor_t, pos_tensor_p, pos_tensor_tr], dim=0)
            
            V = K.clone()

            Q = self.q_fc(Q)
            # print('Q transformed: ', Q.shape) # torch.Size([2048, 512])
            K = self.k_fc(K)
            # print('K transformed: ', K.shape) # torch.Size([4, 512])
            V = self.v_fc(V)

            attention = torch.matmul(Q, K.transpose(1, 0)) / np.sqrt(K.shape[1])
            attention_score[user] = attention.tolist()

        return attention_score


class GRAPH(torch.nn.Module):
    def __init__(self, features, user_features, edge_index, batch_size, num_user, num_item, reg_parm, dim_x, dim_latent=1024, path=None, cluster_dict=None):
        super(GRAPH, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.reg_parm = reg_parm

        self.edge_index = edge_index[:,:2]
        self.edge_index = torch.tensor(self.edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        self.feat = torch.tensor(features, dtype=torch.float).cuda()
        self.user_features = torch.tensor(user_features, dtype=torch.float).cuda()

        self.gnn = GAT(self.feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, dim_latent=dim_latent)

        self.id_embedding = nn.Embedding(num_user+num_item, dim_x)
        nn.init.xavier_normal_(self.id_embedding.weight)

        #self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()

        # read dataset/indices.npy
        # self.indices = np.load(path+'indices.npy', allow_pickle=True)

        # linear layer
        self.MLP_price = MLP_price(dim_in=dim_x*2, dim_out=1)

        # cluster dict
        self.cluster_dict = cluster_dict

        # linear layer (for attention)
        self.q_fc = nn.Linear(dim_x, dim_x, bias=False) # (d_model, d_model)
        self.k_fc = nn.Linear(dim_x, dim_x, bias=False) # (d_model, d_model)
        self.v_fc = nn.Linear(dim_x, dim_x, bias=False) # (d_model, d_model)


    def forward(self, user_nodes, pos_items, neg_items):

        # print('user_nodes: ', user_nodes.shape) # torch.Size([2048])
        # print('pos_items: ', pos_items.shape) # torch.Size([2048])
        # print('neg_items: ', neg_items.shape) # torch.Size([2048])        
        representation = self.gnn(self.id_embedding)

        # self.representation = rep
        # print('tr_rep: ', tr_rep.shape) # torch.Size([num_user+num_item, 512])

        # representation = rep #torch.max_pool2d((v_rep, a_rep, t_rep))#max()#torch.cat((v_rep, a_rep, t_rep), dim=1)
        # print('representation: ', representation.shape) # torch.Size([num_user+num_item, 512])
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]

        # 1) BPR loss
        pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
        neg_score = torch.sum(user_tensor * neg_tensor, dim=1)
        
        # 2) Price loss
        user_pos_tensor = torch.cat((user_tensor, pos_tensor), dim=1)
        # linear layer 
        pred_price = self.MLP_price(user_pos_tensor)
        
        return pos_scores, neg_score, representation, pred_price

    def loss(self, data):
        user, pos_items, neg_items, labels = data
        pos_scores, neg_scores,representation, pred_price = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda())

        # 1) BPR loss
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg = (torch.norm(representation[user])**2
        +torch.norm(representation[pos_items])**2
        +torch.norm(representation[neg_items])**2) / 2
        loss_reg = self.reg_parm*reg/self.batch_size

        loss_BPR = loss_value + loss_reg

        # 2) BCE loss
        loss_Price = F.binary_cross_entropy_with_logits(pred_price.float(), labels.unsqueeze(1).float().cuda())

        return loss_BPR, loss_Price


    def accuracy(self, dataset, indices, topk=10, neg_num=100):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        # bar = tqdm(total=len(dataset))
        recall_list = np.array([0.0, 0.0, 0.0, 0.0])
        ndcg_list = np.array([0.0, 0.0, 0.0, 0.0])


        for data in dataset:
            # bar.update(1)

            sum_item += 1
            user = data[0]
            pos_items = data[1:]
      
            # minus from self.indices to pos_items
            neg_items = [x for x in indices if x not in pos_items]
            neg_items = neg_items[:neg_num]
            neg_items = list(neg_items)

            batch_user_tensor = torch.tensor(user).cuda() 
            batch_pos_tensor = torch.tensor(pos_items).cuda()
            batch_neg_tensor = torch.tensor(neg_items).cuda()

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

             # make the topk items using the price score

            pos_price = self.MLP_price(torch.cat((user_embed.repeat(len(pos_items),1), pos_v_embed), dim=1))
            neg_price = self.MLP_price(torch.cat((user_embed.repeat(neg_num,1), neg_v_embed), dim=1))


            # make the topk items using the rank score
            # save the result from topk 5 to topk 10
            for k, topk in enumerate(range(5, 21, 5)):
                _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
                index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
                
                num_hit = len(index_set.difference(all_set))
                # sum_pre += float(num_hit/topk)
      
                recall = float(num_hit/num_pos)
                ndcg_score = 0.0
                for i in range(num_pos):
                    label_pos = neg_num + i
                    if label_pos in index_of_rank_list:
                        index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                        ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
                ndcg = ndcg_score/num_pos

                # append to the list
                recall_list[k] += recall
                ndcg_list[k] += ndcg

        return recall_list/sum_item, ndcg_list/sum_item


class GCN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_id, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features

        self.preference = nn.Embedding(num_user, self.dim_latent) 
        nn.init.xavier_normal_(self.preference.weight).cuda()
        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            self.conv_embed_1 = GCNConv(self.dim_latent, self.dim_latent, aggr='add')
            # nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GCNConv(self.dim_feat, self.dim_feat, aggr='add')
            # nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = GCNConv(self.dim_id, self.dim_id, aggr='add')
        # nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
        nn.init.xavier_normal_(self.g_layer2.weight)

    def forward(self, id_embedding):
        temp_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features
        x = torch.cat((self.preference.weight, temp_features), dim=0)
        x = F.normalize(x).cuda()
 
        #layer-1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)

        return x_1
        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        x = torch.cat((x_1, x_2), dim=1)

        return x


class GAT(torch.nn.Module):
    def __init__(self, features, user_features, edge_index, batch_size, num_user, num_item, dim_id, DROPOUT, dim_latent=None):
        super(GAT, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.user_dim_feat = user_features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.user_features = user_features

        # self.preference = nn.Embedding(num_user, self.dim_latent)
        # nn.init.xavier_normal_(self.preference.weight).cuda()
        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.user_MLP = nn.Linear(self.user_dim_feat, self.dim_latent)

            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, DROPOUT, aggr='add')
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
        nn.init.xavier_normal_(self.g_layer2.weight)

    def forward(self, id_embedding):
        temp_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features
        # temp_features = nn.Embedding.from_pretrained(self.features, freeze=True).weight
        temp_user_features = torch.tanh(self.user_MLP(self.user_features)) if self.dim_latent else self.user_features

        x = torch.cat((temp_features, temp_user_features), dim=0)
        x = F.normalize(x).cuda()

        #layer-1
        # print('x',x.max())
        # print('edge_index',self.edge_index.max())

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)

        return x_1

        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        x = torch.cat((x_1, x_2), dim=1)

        return x

# define a MLP class with 2 linear layers
class MLP_price(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_price, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear_layer1 = nn.Linear(self.dim_in, self.dim_in//2)
        nn.init.xavier_normal_(self.linear_layer1.weight)
        self.linear_layer2 = nn.Linear(self.dim_in//2, self.dim_out)
        nn.init.xavier_normal_(self.linear_layer2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.linear_layer1(x))
        x = torch.sigmoid(self.linear_layer2(x))

        return x

