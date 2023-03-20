import os
import argparse
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoad import DataLoad
from Model import MMMO, GRAPH
import pickle
import matplotlib.pyplot as plt
import random

class Net:
    def __init__(self, args):
        np.random.seed(args.seed)
        random.seed(args.seed)
        self.model_name = args.model_name
        self.collection = args.collection
        self.data_path = args.data_path
        self.PATH_weight_load = args.PATH_weight_load
        self.PATH_weight_save = args.PATH_weight_save
        self.l_r = args.l_r
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.dim_latent = args.dim_latent
        self.num_epoch = args.num_epoch
        self.num_workers = args.num_workers
        self.reg_parm = args.reg_parm
        self.neg_sample = args.neg_sample
        self.loss_alpha = args.loss_alpha
        self.number = args.number
        self.attention_dropout = args.attention_dropout
        self.device = "cuda:0" # torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.dim_v = 1024
        self.dim_t = 1500
        self.dim_p = 64
        self.dim_tr = 64
        self.patience = 10 # for early stopping

        # data path: 'dataset/collections/bayc'
        self.data_path = os.path.join(self.data_path, self.collection)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        print(self.data_path)

        # save path: 'saved/MMMO/bayc'
        save_path = os.path.join(self.PATH_weight_save, self.model_name, self.collection)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(save_path)
            
        
        # save hyperparameters
        hyperparameters = {'model_name': self.model_name, 'data_path': self.data_path, 'learning_rate': self.l_r, 
                            'dim_latent': self.dim_latent, 'reg_parm': self.reg_parm, 'loss_alpha': self.loss_alpha}
        with open(save_path + f'/{str(self.number)}' + '_hyperparameters' + '.pickle', 'wb') as f:
            pickle.dump(hyperparameters, f)

        ###################################### Load data ######################################
        print('Loading data  ...')
        num_user_item = np.load(self.data_path + 'num_user_item.npy', allow_pickle=True).item()
        self.num_item = num_user_item['num_item']
        self.num_user = num_user_item['num_user']
        print(f"num_user: {self.num_user}, num_item: {self.num_item}")

        self.train_dataset = DataLoad(self.data_path, self.num_user, self.num_item, 0)
        for i in range(1, self.neg_sample+1): 
            self.train_dataset += DataLoad(self.data_path, self.num_user, self.num_item, i)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

        self.edge_index = np.load(self.data_path + 'train.npy')
        self.val_dataset = np.load(self.data_path + 'val.npy', allow_pickle=True)
        self.test_dataset = np.load(self.data_path + 'test.npy', allow_pickle=True)

        self.v_feat = np.load(self.data_path + 'image_feat.npy')
        self.t_feat = np.load(self.data_path + 'text_feat.npy')
        self.p_feat = np.load(self.data_path + 'price_feat.npy')
        self.tr_feat = np.load(self.data_path + 'transaction_feat.npy')

        self.user_feat = np.load(self.data_path + 'user_feat.npy') 

        self.indices_valid = np.load(self.data_path + 'indices_valid.npy', allow_pickle=True)
        self.indices_test = np.load(self.data_path + 'indices_test.npy', allow_pickle=True)

        self.cluster_dict = np.load(self.data_path + 'cluster_dict.npy', allow_pickle=True).item()

        ###################################### Load model ######################################
        print('Loading model  ...')
        if self.model_name == 'MMMO':
            self.features = [self.v_feat, self.t_feat, self.p_feat, self.tr_feat]
            self.model = MMMO(self.features, self.user_feat, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.reg_parm, self.dim_latent, self.attention_dropout, self.data_path).cuda()

        elif self.model_name == 'GRAPH':
            self.features = np.concatenate((self.v_feat, self.t_feat, self.p_feat, self.tr_feat), axis=1)
            print(self.features.shape)
            self.model = GRAPH(self.features, self.user_feat, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.reg_parm, self.dim_latent,self.data_path).cuda()

        # elif self.model_name == 'MGCN':
        #     self.features = [self.v_feat, self.a_feat, self.t_feat]
        #     self.model = MGCN(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item,
        #                        self.dim_latent,self.data_path).cuda()

        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.l_r}],
                                          weight_decay=self.weight_decay)
        
        if self.PATH_weight_load and os.path.exists(self.PATH_weight_load):
            self.model.load_state_dict(torch.load(self.PATH_weight_load))
            print('module weights loaded....')


    def run(self):

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        losses, losses_BPR, losses_Price = [], [], []
        recalls, ndcgs, recalls_test, ndcgs_test = [], [], [], []
        best_recall, best_ndcg = 0.0, 0.0
        
        i = 8
        print(f'{self.data_path[i:-1]} : Start training ...')
        patience_count = 0
        for epoch in range(self.num_epoch):

            ###################################### Train ######################################
            self.model.train()
            loss_batch = 0.0
            loss_BPR_batch = 0.0
            loss_Price_batch = 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                loss_BPR, loss_Price = self.model.loss(data)
                # print('loss_BPR:',  loss_BPR.item(), ' loss_Price:', loss_Price.item())
                self.loss = (1-self.loss_alpha)*loss_BPR + self.loss_alpha*loss_Price
                self.loss.backward()
                self.optimizer.step()

                loss_batch += self.loss
                loss_BPR_batch += loss_BPR
                loss_Price_batch += loss_Price

            loss_avg = loss_batch.item() / self.batch_size
            loss_BPR_avg = loss_BPR_batch.item() / self.batch_size
            loss_Price_avg = loss_Price_batch.item() / self.batch_size

            ###################################### Evaluate ######################################
            self.model.eval()
            with torch.no_grad():
                # Valid
                recall, ndcg = self.model.accuracy(self.val_dataset, self.indices_valid)
                recall, ndcg = recall[1], ndcg[1] # top10만 보기
                # Test
                recall_test, ndcg_test = self.model.accuracy(self.test_dataset, self.indices_test)
                recall_test, ndcg_test = recall_test[1], ndcg_test[1] # top10만 보기

            # Best model 저장
            if recall > best_recall: 
                patience_count = 0
                best_recall, best_ndcg = recall, ndcg
                torch.save(self.model.state_dict(), self.PATH_weight_save+self.model_name+ self.data_path[i:-1] + str(self.number)  +'.pt')

                if self.model_name == 'MGAT' or self.model_name == 'MGAT_2':
                    attention_score = self.model.attention_score(self.test_dataset)
                    with open(self.PATH_weight_save + self.model_name + self.data_path[i:-1] + str(self.number) + '_attention_score' + '.pickle', 'wb') as f:
                        pickle.dump(attention_score, f)
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break
            
            ###################################### Epoch append ######################################
            print(
                '{0}-th Loss:{1:.4f}, BPR:{7:.4f} Price: {8:.4f} / Recall:{2:.4f} NDCG:{3:.4f} / Best: Recall:{4:.4f} NDCG:{5:.4f} / collection: {6}'.format(
                    epoch, loss_avg, recall, ndcg, best_recall, best_ndcg, self.data_path[i:-1], loss_BPR_avg, loss_Price_avg))
 
            losses.append(loss_avg)
            losses_BPR.append(loss_BPR_avg)
            losses_Price.append(loss_Price_avg)

            recalls.append(recall)
            ndcgs.append(ndcg)

            recalls_test.append(recall_test)
            ndcgs_test.append(ndcg_test)


        ###################################### Epoch 저장 ######################################
        print('Best Recall:{0:.4f} NDCG:{1:.4f}'.format(best_recall, best_ndcg))

        result = {f'losses': losses, f'losses_BPR': losses_BPR, f'losses_Price': losses_Price, 
                  f'recalls': recalls, f'ndcgs': ndcgs, 
                  f'recalls_test': recalls_test, f'ndcgs_test': ndcgs_test}
        with open(self.PATH_weight_save + self.model_name + self.data_path[i:-1] + str(self.number) + '_losses' + '.pickle', 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MMMO', help='Model name.')
    parser.add_argument('--collection', default='bayc', help='Collection name.')
    parser.add_argument('--data_path', default='dataset/collections/', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default='saved', help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number')
    parser.add_argument('--reg_parm', type=float, default=0.001, help='Workers number')
    parser.add_argument('--neg_sample', type=int, default=5, help='num of negative samples for training')
    parser.add_argument('--loss_alpha', type=float, default=0, help='alpha for loss')
    parser.add_argument('--number', type=int, default=0, help='구분을 위한 숫자')
    parser.add_argument('--attention_dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--seed', type=float, default=2023, help='Random seed')
    args = parser.parse_args()

    mgat = Net(args)
    mgat.run()
