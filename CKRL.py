import os
import numpy as np
import re
import pickle as pk
import random
from numpy import linalg as LA

from CorrelationEmbedding import *
from InputDataStructure import *
from DataPreproccessing import *
from DataModel import *
from ModelEvaluator import *

train_path = r'Dataset\FB15K-237\train.txt'
test_path = r'Dataset\FB15K-237\test.txt'

save_input_data_path = r'Data\dataset'
save_preproccessing_data_path = r'Data\preproccessed_data'
save_model_path = r'Data\data_model.dat'
evaluation_result_path = r'Data\evaluation.dat'

CE = CorrelationEmbedding()

embedding_size = 50

CE.load_datesets(train_path, test_path, save_input_data_path)
CE.preproccess(save_preproccessing_data_path, embedding_size)

model = CE.data_model(save_model_path, embedding_size)

max_iter = 1000

for cur_iter in range(max_iter):
    print(f'{cur_iter}/{max_iter}')
    model.train()