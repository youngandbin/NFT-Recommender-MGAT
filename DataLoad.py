import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DataLoad(Dataset):
	def __init__(self, path, num_user, num_item, pop_num):
		super(DataLoad, self).__init__()
		self.data = np.load(path+'train.npy', allow_pickle=True)
		self.pop = np.load(path+'indices_valid.npy', allow_pickle=True)
		# self.adj_lists = np.load(path+'final_adj_dict.npy').item()
		# self.all_set = set(range(num_user, num_user+num_item))
		self.adj_lists = np.load(path+'adj_dict.npy', allow_pickle=True).item() # dic형태를 가져오려면 item 함수 사용
		self.all_set = set(range(num_item))
		self.pop_num = pop_num

	def __getitem__(self, index):
		user, pos_item, labels = self.data[index]
		neg_item = [x for x in self.pop if x not in self.adj_lists[user]][self.pop_num]
		#neg_item = self.all_set.difference(self.adj_lists[user])
		return [user, pos_item, neg_item, labels]

	def __len__(self):
		return len(self.data)
