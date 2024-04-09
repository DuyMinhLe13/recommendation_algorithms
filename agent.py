import numpy as np
import pandas as pd
import scipy
import os
import zipfile
from tqdm import tqdm
from fpgrowth import fpgrowth

import os
os.environ['KAGGLE_USERNAME'] = "minhduyl"
os.environ['KAGGLE_KEY'] = "13b464371b84cde3768570bd8d15e1a6"
import kaggle
kaggle.api.authenticate()

class Agent():
    def __init__(self, dataset_path='dataset', weight_path='weight', download_dataset=True, download_weight=True):

        if not os.path.exists(dataset_path) and download_dataset:
            kaggle.api.dataset_download_files('hernan4444/anime-recommendation-database-2020', path=dataset_path, unzip=True, quiet=False)

        if not os.path.exists(weight_path) and download_weight:
            kaggle.api.dataset_download_files('duyminhle/anime-recommendation-system-weight', path=weight_path, unzip=True, quiet=False)

        self.weight = np.load(weight_path + '/weight.npy')
        self.anime_index = np.load(weight_path + '/anime_index.npy')
        self.anime_df = pd.read_csv(dataset_path + '/anime.csv')
        self.anime_df = self.anime_df.loc[self.anime_df['MAL_ID'].isin(self.anime_index)]
        self.anime_df = self.anime_df.sort_values(by=['MAL_ID'])
        self.episode_embedding = scipy.sparse.load_npz(weight_path + '/episode_embedding.npz')
        self.user_index = np.load(weight_path + '/user_index.npy')
        self.anime_rating_embedding = scipy.sparse.load_npz(weight_path + '/anime_rating_embedding.npz')
        self.user_item_matrix = self.anime_rating_embedding.transpose()

    def build_itemSetList(self, num_users=20000, num_animes=1000):
        dataset = self.user_item_matrix[:num_users, :num_animes]
        nonzero_indices = dataset.nonzero()
        nonzero_indices = np.concatenate((nonzero_indices[0].reshape(1, -1), nonzero_indices[1].reshape(1, -1)), axis=0).transpose()
        self.itemSetList = np.split(nonzero_indices[:,1], np.unique(nonzero_indices[:, 0], return_index=True)[1][1:])
        self.itemSetList = list(map(lambda x: self.anime_index[x].tolist(), self.itemSetList))

    def build_fpgrowth(self, minSup=0.19, minConf=0.5):
        self.freqItemSet_fpgrowth, self.rules_fpgrowth = fpgrowth(self.itemSetList, minSupRatio=minSup, minConf=minConf)

    def find_similar_animes(self, id: int = None, name: str = None, k=10, return_df=False):
        if isinstance(id, int):
            index = self.anime_df[self.anime_df.MAL_ID == id].index[0]
        elif isinstance(name, str):
            index = self.anime_df[self.anime_df.Name == name].index[0]
        else: raise Exception('id or name arguments not suitable, type(id) is int or type(name) is str')

        index_res = (-self.weight[index]).argsort()[:k]

        if return_df: return self.anime_df.loc[index_res]

        if isinstance(id, int):
            return self.anime_df.loc[index_res].MAL_ID.tolist()
        return self.anime_df.loc[index_res].Name.tolist()

    def find_anime_for_user_using_episode(self, id: int = None, top_k=5, num_animes=2, return_df=False, return_name=False):
        if isinstance(id, int):
            index = np.where(self.user_index == id)[0][0]
        else: raise Exception('id or name arguments not suitable, type(id) is int or type(name) is str')

        user_episode_data = np.array(self.episode_embedding[index].todense())[0]
        anime_indexes = (-user_episode_data).argsort()[:top_k]

        index_res = []
        for anime_index in anime_indexes:
            index_res += self.find_similar_animes(id=int(self.anime_index[int(anime_index)]), k=num_animes)
        
        if return_df:
            return self.anime_df.loc[self.anime_df['MAL_ID'].isin(index_res)]
    
        if return_name:
            return self.anime_df.loc[self.anime_df['MAL_ID'].isin(index_res)]['Name']

        return index_res
    
    def find_anime_for_user_using_rating(self, id: int, top_k=5, num_animes=2, return_df=False, return_name=False):
        if isinstance(id, int):
            index = np.where(self.user_index == id)[0][0]
        else: raise Exception('id or name arguments not suitable, type(id) is int or type(name) is str')

        user_rating_data = np.array(self.anime_rating_embedding[:, index].todense())[:, 0]
        user_rating_predict = user_rating_data @ self.weight
        anime_indexes = (-user_rating_predict).argsort()[:top_k]

        index_res = []
        for anime_index in anime_indexes:
            index_res += self.find_similar_animes(id=int(self.anime_index[int(anime_index)]), k=num_animes)
        
        if return_df:
            return self.anime_df.loc[self.anime_df['MAL_ID'].isin(index_res)]
        
        if return_name:
            return self.anime_df.loc[self.anime_df['MAL_ID'].isin(index_res)]['Name']
        return index_res
    
    def find_anime_for_user_using_fpgrowth(self, id: int, return_df=False, return_name=False):
        if isinstance(id, int):
            index = np.where(self.user_index == id)[0][0]
        else: raise Exception('id arguments not suitable, type(id) is int')
        anime_set = set(self.anime_index[np.where(self.user_item_matrix[index].toarray() > 0)[1]])
        index_res = set()
        for rule in self.rules_fpgrowth:
            if rule[0].issubset(anime_set):
                index_res = index_res.union(rule[1])
        
        index_res = list(index_res)
        
        if return_df:
            return self.anime_df.loc[self.anime_df['MAL_ID'].isin(index_res)]
        
        if return_name:
            return self.anime_df.loc[self.anime_df['MAL_ID'].isin(index_res)]['Name']
        return index_res
    
