import os
import numpy as np
import re
import pickle as pk
import random
from numpy import linalg as LA

from InputDataStructure import *


class DataPreproccessing:

    def __init__(self, data_structure: InputDataStructure, embedding_dimension, save_data_path):

        self.data_structure = data_structure
        self.embedding_dimension = embedding_dimension

        if not self.load(save_data_path):

            self.dic_positive_samples = self.positive_data()
            self.dic_negative_samples = self.negative_data(self.dic_positive_samples)
            self.dic_positive_embedding = self.positive_embedding(self.dic_positive_samples)
            self.dic_negative_embedding = self.negative_embedding(self.dic_negative_samples)

            self.entity_embeddings = self.init_entity_embeddings()
            self.dic_relation_embeddings = self.init_relation_embeddings()

            self.save(save_data_path)

    def save(self, save_data_path):
        pass
        #with open(save_data_path, 'wb') as f:
        #    pk.dump(self, f, pk.HIGHEST_PROTOCOL)

    def load(self, save_data_path):

        if not os.path.isfile(save_data_path):
            return False

        else:
            with open(save_data_path, 'rb') as f:
                data = pk.load(f)
                self.dic_positive_samples = data.dic_positive_samples
                self.dic_negative_samples = data.dic_negative_samples
                self.dic_positive_embedding = data.dic_positive_embedding
                self.dic_negative_embedding = data.dic_negative_embedding
                self.entity_embeddings = data.entity_embeddings
                self.dic_relation_embeddings = data.dic_relation_embeddings
            return True

    def positive_data(self):

        dic_positive_samples = dict()
        for h, r, t in self.data_structure.triples_id:

            if r in dic_positive_samples:
                dic_positive_samples[r].append((h, t))
            else:
                dic_positive_samples[r] = [(h, t)]

        for r in dic_positive_samples.keys():
            dic_positive_samples[r] = np.array(dic_positive_samples[r])

        return dic_positive_samples

    def negative_data(self, dic_positive_data):

        dic_negative_samples = dict()
        for relation, pair_list in dic_positive_data.items():

            for h, t in pair_list:

                hp, tp = self.select_pair(h, t)
                if relation in dic_negative_samples:

                    dic_negative_samples[relation].append((hp, t))
                    dic_negative_samples[relation].append((h, tp))

                else:
                    dic_negative_samples[relation] = [(hp, t)]
                    dic_negative_samples[relation].append((h, tp))


        for r in dic_negative_samples.keys():
            dic_negative_samples[r] = np.array(dic_negative_samples[r])

        return dic_negative_samples

    def positive_embedding(self,  dic_positive_data):

        dic_positive_embedding = dict()
        for relation, pairList in dic_positive_data.items():

            H = np.random.randn(len(pairList), self.embedding_dimension)
            T = np.random.randn(len(pairList), self.embedding_dimension)
            dic_positive_embedding[relation] = (H, T)

        return dic_positive_embedding

    def negative_embedding(self,  dic_negative_data):
        
        dic_negative_embedding = dict()
        for relation, NegpairList in dic_negative_data.items():

            Hp = np.random.randn(len(NegpairList), self.embedding_dimension)
            Tp = np.random.randn(len(NegpairList), self.embedding_dimension)
            dic_negative_embedding[relation] = (Hp, Tp)

        return dic_negative_embedding

    def select_pair(self, h, t):

        while True:

            hp = self.data_structure.dic_entity_id[random.choice(
                self.data_structure.entities)]
            tp = self.data_structure.dic_entity_id[random.choice(
                self.data_structure.entities)]
            if h != hp and t != tp and hp != t and h != tp:
                break

        return hp, tp

    def init_entity_embeddings(self):

        entity_count = len(self.data_structure.entities)
        embeddings = np.random.rand(entity_count, self.embedding_dimension)
        return embeddings

    def init_relation_embeddings(self):
        
        dic_relation_embedding = dict()

        for id in self.data_structure.dic_id_relation.keys():
            embeddings = dict()
            embeddings['H'] = np.random.rand(self.embedding_dimension * 2, self.embedding_dimension)
            embeddings['T'] = np.random.rand(self.embedding_dimension * 2, self.embedding_dimension)
            embeddings['Hp'] = np.random.rand(self.embedding_dimension * 2, self.embedding_dimension)
            embeddings['Tp'] = np.random.rand(self.embedding_dimension * 2, self.embedding_dimension)
            dic_relation_embedding[id] = embeddings

        return dic_relation_embedding