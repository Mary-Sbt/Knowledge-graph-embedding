import os
import numpy as np
import re
import pickle as pk
import random
from numpy import linalg as LA

from InputDataStructure import *
from DataPreproccessing import *
from DataModel import *
from ModelEvaluator import *

class CorrelationEmbedding:

    def __init__(self):

        self.train_dataset = None
        self.test_dataset = None

    def load_input_file(self, path, save_data_path):

        triples = []
        with open(path, 'r') as f:

            # triples = [(/m/027rn,/location/country/form_of_government,/m/06cx9), (), ...]
            for line in f.readlines():
                h, r, t = re.split('\t', line.lower().strip())
                triples.append((h, r, t))
                #h, *rs, t = re.split('[\t.]', line.lower().strip())
                #triples.extend((h, r, t) for r in rs)

        dataset = InputDataStructure(triples, save_data_path)
        return dataset
       
    def load_datesets(self, train_path, test_path, save_data_path):

        self.train_dataset = self.load_input_file(train_path, save_data_path + '.train.dat')
        self.test_dataset  = self.load_input_file(test_path, save_data_path + '.test.dat')

    def preproccess(self, save_data_path, embedding_dimension):

        self.train_preproccessed_data = DataPreproccessing(
            self.train_dataset, embedding_dimension, save_data_path + '.train.dat')
        if self.test_dataset != None:
            self.test_preproccessed_data = DataPreproccessing(
                self.test_dataset, embedding_dimension, save_data_path + '.test.dat')

    def data_model(self, save_data_path, latent_dimension):

        self.solve_tranfser_matrices = DataModel(
            self.train_dataset, self.train_preproccessed_data, latent_dimension, save_data_path)

        return self.solve_tranfser_matrices