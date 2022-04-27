import os
import numpy as np
import re
import pickle as pk
import random
from numpy import linalg as LA
from scipy.spatial.distance import cosine

from InputDataStructure import *
from DataPreproccessing import *
from DataModel import *

class ModelEvaluator:

    def __init__(self, train_data:DataPreproccessing, test_data:DataPreproccessing, model:DataModel, save_path):
        
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.save_path = save_path

    def evaluate(self, evaluate_test):

        rs = self.train_data.dic_positive_embedding.keys()

        self.hist_data = []

        for j, r in enumerate(rs):

            if not evaluate_test:
                iHT =  self.train_data.dic_positive_samples[r]
            elif r in self.test_data.dic_positive_samples:
                iHT =  self.test_data.dic_positive_samples[r]
            else:
                continue

            _, score = self.evaluate_relation(r, iHT)
            self.hist_data.append(score)

        if evaluate_test: print(f'mean score (test)={np.mean(self.hist_data)}')
        else: print(f'mean score (train)={np.mean(self.hist_data)}')

        with open(self.save_path, 'wb') as f:
            pk.dump(self, f, pk.HIGHEST_PROTOCOL)

    def evaluate_relation(self, r, iHT):

        iH, iT = iHT[:,0], iHT[:,1]

        H_entity_embeddings = self.train_data.entity_embeddings[iH]
        T_entity_embeddings = self.train_data.entity_embeddings[iT]

        entity_pair_embeddings = np.c_[H_entity_embeddings, T_entity_embeddings]

        H_relation_embedding = self.train_data.dic_relation_embeddings[r]['H']
        T_relation_embedding = self.train_data.dic_relation_embeddings[r]['T']

        H = self.map_entity_embedding_to_pair_embeddings(entity_pair_embeddings, H_relation_embedding)
        T = self.map_entity_embedding_to_pair_embeddings(entity_pair_embeddings, T_relation_embedding)

        n, m = H.shape

        Wh, Wt = self.model.mapping_to_correlation_space_matrices[r]

        LT = np.dot(T, Wt)
        LH = np.dot(H, Wh)

        pos_corrs, neg_corrs = [], []

        pos_corrs = self.correlation(LH, LT)
        mean_corr_score = sum(pos_corrs) / n

        return pos_corrs, mean_corr_score
    
    def correlation(self, A, B):
        a_n = np.linalg.norm(A, axis=1, keepdims=True)
        b_n = np.linalg.norm(B, axis=1, keepdims=True)
        A = A / a_n
        B = B / b_n
        cosine_similarity = np.sum(A * B,axis=1)
        return cosine_similarity

    def map_entity_embedding_to_pair_embeddings(self, entity_pair_embeddings, relation_embedding):
        E = entity_pair_embeddings
        R = relation_embedding
        A = E @ R
        return A
