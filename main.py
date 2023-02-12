import os
import argparse
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoad import DataLoad
from Model import MMMO, GRAPH
from Model_2 import MMMO_2 # 2-hop version
import pickle
import matplotlib.pyplot as plt
import random

class Net:
    def __init__(self, args):
        self.patience = 10
        ##########################################################################################################################################
        # seed = args.seed
        # np.random.seed(seed)
        # random.seed(seed)
        self.device = "cuda:0" #torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        ##########################################################################################################################################
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.dim_latent = args.dim_latent
        self.reg_parm = args.reg_parm
        self.neg_sample = args.neg_sample
        self.dim_v = 1024
        self.dim_t = 1500
        self.dim_p = 64
        self.dim_tr = 64
        self.loss_alpha = args.loss_alpha
        self.number = str(args.number)
        # print('args: ', args)
        # save the hyperparameters in a file
        hyperparameters = {'model_name': args.model_name, 'data_path': args.data_path, 'learning_rate': args.l_r, 
                            'dim_latent': args.dim_latent, 'reg_parm': args.reg_parm, 'loss_alpha': args.loss_alpha}
        with open(args.PATH_weight_save + args.model_name + args.data_path[8:-1] + self.number + '_hyperparameters' + '.pickle', 'wb') as f:
            pickle.dump(hyperparameters, f)

        ##########################################################################################################################################
        print('Loading data  ...')
        # self.num_item = df[:,1].max() + 1
        # self.num_user = df[:,0].max() - self.num_item + 1
        num_user_item = np.load(self.data_path + 'num_user_item.npy', allow_pickle=True).item()

        self.num_item = num_user_item['num_item']
        self.num_user = num_user_item['num_user']

        print(f"num_user: {self.num_user}, num_item: {self.num_item}")

        self.train_dataset = DataLoad(self.data_path, self.num_user, self.num_item, 0)

        for i in range(1, self.neg_sample):
            self.train_dataset += DataLoad(self.data_path, self.num_user, self.num_item, i)

        # self.train_dataset = DataLoad(self.data_path, self.num_user, self.num_item)
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

        self.indices_test = np.load(self.data_path + 'indices_test.npy', allow_pickle=True)
        self.indices_valid = np.load(self.data_path + 'indices_valid.npy', allow_pickle=True)

        self.cluster_dict = np.load(self.data_path + 'cluster_dict.npy', allow_pickle=True).item()

        print('Data has been loaded.')
        ##########################################################################################################################################
        if self.model_name == 'MGAT':
            self.features = [self.v_feat, self.t_feat, self.p_feat, self.tr_feat]
            self.model = MMMO(self.features, self.user_feat, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.reg_parm, self.dim_latent, args.attention_dropout, args.data_path).cuda()
        if self.model_name == 'MGAT_2':
            self.features = [self.v_feat, self.t_feat, self.p_feat, self.tr_feat]
            self.model = MMMO_2(self.features, self.user_feat, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.reg_parm, self.dim_latent, args.attention_dropout, args.data_path).cuda()

        elif self.model_name == 'GRAPH':
            self.features = np.concatenate((self.v_feat, self.t_feat, self.p_feat, self.tr_feat), axis=1)
            print(self.features.shape)
            self.model = GRAPH(self.features, self.user_feat, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.reg_parm, self.dim_latent,args.data_path).cuda()

        elif self.model_name == 'MGCN':
            self.features = [self.v_feat, self.a_feat, self.t_feat]
            # self.model = MGCN(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item,
            #                    self.dim_latent,args.data_path).cuda()


        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}],
                                          weight_decay=self.weight_decay)
        ##########################################################################################################################################
        
    def run(self):

        fix_seed = 2023
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        best_precision, best_recall, best_ndcg_score = 0., 0.0, 0.0
        i = 8
        print(f'{args.data_path[i:-1]} : Start training ...')
        epoches = np.arange(self.num_epoch)
        losses = []
        BPR_losses = []
        Price_losses = []
        recalls = []
        ndcgs = []
        
        patience_count = 0

        for epoch in range(self.num_epoch):
            self.model.train()
            sum_loss = 0.0
            bpr_loss = 0.0
            price_loss = 0.0
            for data in self.train_dataloader:
                # multi-objective loss로 학습
                self.optimizer.zero_grad()
                BPR_loss, Price_loss = self.model.loss(data)
                # print('BPR_loss:',  BPR_loss.item(), ' Price_loss:', Price_loss.item())
                alpha=self.loss_alpha
                self.loss = (1-alpha)*BPR_loss + alpha*Price_loss
                self.loss.backward()
                self.optimizer.step()
                sum_loss += self.loss
                bpr_loss += BPR_loss
                price_loss += Price_loss

            current_loss = sum_loss.item() / self.batch_size
            current_bpr_loss = bpr_loss.item() / self.batch_size
            current_price_loss = price_loss.item() / self.batch_size

            self.model.eval()
            with torch.no_grad():
                # 5, 10, 15, 20 순서 topk10 기준으로 best model 고르기
                recall, ndcg_score = self.model.accuracy(self.val_dataset, self.indices_valid)
                recall = recall[1]
                ndcg_score = ndcg_score[1]
            if recall > best_recall: 
                patience_count = 0
                best_recall, best_ndcg_score = recall, ndcg_score
                torch.save(self.model.state_dict(), args.PATH_weight_save+args.model_name+ args.data_path[i:-1] + self.number  +'.pt')
                # testdataset 결과 저장
                test_recall, test_ndcg_score = self.model.accuracy(self.test_dataset, self.indices_test)
                
                if args.model_name == 'MGAT' or args.model_name == 'MGAT_2':
                    # save final attention score
                    attention_score = self.model.attention_score(self.test_dataset)
                    with open(args.PATH_weight_save + args.model_name + args.data_path[i:-1] + self.number + '_attention_score' + '.pickle', 'wb') as f:
                        pickle.dump(attention_score, f)
                
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

            print(
                '{0}-th Loss:{1:.4f}, BPR:{7:.4f} Price: {8:.4f} / Recall:{2:.4f} NDCG:{3:.4f} / Best: Recall:{4:.4f} NDCG:{5:.4f} / collection: {6}'.format(
                    epoch, current_loss, recall, ndcg_score, best_recall, best_ndcg_score, args.data_path[i:-1], current_bpr_loss, current_price_loss))
 
            losses.append(current_loss)
            BPR_losses.append(current_bpr_loss)
            Price_losses.append(current_price_loss)
            recalls.append(recall)
            ndcgs.append(ndcg_score)

        print('Best Recall:{0:.4f} NDCG:{1:.4f}'.format(best_recall, best_ndcg_score))

        # save losses, BPR_losses, Price_losses, recalls, ndcgs as pickle file
        result = {f'losses': losses, f'BPR_losses': BPR_losses, f'Price_losses': Price_losses, f'recalls': recalls, f'ndcgs': ndcgs}
        with open(args.PATH_weight_save + args.model_name + args.data_path[i:-1] + self.number + '_losses' + '.pickle', 'wb') as f:
            pickle.dump(result, f)

        # save result as pickle file
        result = {f'Best_Recall': best_recall, f'Best_NDCG': best_ndcg_score}
        with open(args.PATH_weight_save + args.model_name + args.data_path[i:-1] + self.number +'_best'+ '.pickle', 'wb') as f:
            pickle.dump(result, f)

        # save result as pickle file
        result = {f'Best_Recall': test_recall, f'Best_NDCG': test_ndcg_score}
        with open(args.PATH_weight_save + args.model_name + args.data_path[i:-1] + self.number + '_test' + '.pickle', 'wb') as f:
            pickle.dump(result, f)


  

if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MGAT', help='Model name.')
    parser.add_argument('--data_path', default='dataset/collections/bayc/', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default='saved/', help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number')
    parser.add_argument('--reg_parm', type=float, default=0.001, help='Workers number')
    parser.add_argument('--neg_sample', type=int, default=5, help='negative sample ratio')
    parser.add_argument('--loss_alpha', type=float, default=0, help='alpha for loss')
    parser.add_argument('--number', type=int, default=0, help='구분을 위한 숫자')
    parser.add_argument('--attention_dropout', type=float, default=0.2, help='dropout ratio')
    args = parser.parse_args()

    mgat = Net(args)
    mgat.run()
