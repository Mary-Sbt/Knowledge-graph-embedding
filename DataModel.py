import os
import numpy as np
import re
import pickle as pk
import random
from numpy import linalg as LA
from sklearn.preprocessing import normalize

from InputDataStructure import *
from DataPreproccessing import *

class DataModel:

    def __init__(self, data_structure: InputDataStructure, data_preprocessing: DataPreproccessing, latent_dimension, save_model_path):
        
        if not self.load(save_model_path):
                
            self.data_structure = data_structure
            self.data_preprocessing = data_preprocessing
            self.latent_dimension = latent_dimension
            self.save_model_path = save_model_path
            self.mapping_to_correlation_space_matrices = dict()

    def correlation(self, A, B):
        a_n = np.linalg.norm(A, axis=1, keepdims=True)
        b_n = np.linalg.norm(B, axis=1, keepdims=True)
        A = A / a_n
        B = B / b_n
        cosine_similarity = np.sum(A * B,axis=1)
        return cosine_similarity
  
    def train(self):

        dim = self.latent_dimension
        relation_entity_embeddings = dict()

        for r in self.data_preprocessing.dic_positive_embedding.keys():
            H, T =  self.data_preprocessing.dic_positive_embedding[r]
            Hp, Tp =  self.data_preprocessing.dic_negative_embedding[r]
            
            Wh, Wt = self.solve_tranfser_matrices_cca(H, T, Hp, Tp, dim)
            H, T, Hp, Tp = self.solve_embeddings(Wh, Wt, H, T, Hp, Tp)
            H, T, Hp, Tp = self.normalize(H, T, Hp, Tp)

            self.mapping_to_correlation_space_matrices[r] = (Wh, Wt)

            iHT =  self.data_preprocessing.dic_positive_samples[r]
            iHpTp =  self.data_preprocessing.dic_negative_samples[r]
            
            iH, iT = iHT[:,0], iHT[:,1]
            iHp, iTp = iHpTp[:,0], iHpTp[:,1]

            H_entity_embeddings = self.data_preprocessing.entity_embeddings[iH]
            T_entity_embeddings = self.data_preprocessing.entity_embeddings[iT]
            Hp_entity_embeddings = self.data_preprocessing.entity_embeddings[iHp]
            Tp_entity_embeddings = self.data_preprocessing.entity_embeddings[iTp]
            
            head_entity_pair_embeddings = np.r_[H_entity_embeddings, Hp_entity_embeddings, H_entity_embeddings]
            tail_entity_pair_embeddings = np.r_[T_entity_embeddings, Tp_entity_embeddings, T_entity_embeddings]

            head_pair_embeddings = np.r_[H, Hp, H]
            tail_pair_embeddings = np.r_[T, Tp, T]

            H_relation_embedding = self.least_square(head_entity_pair_embeddings, head_pair_embeddings)
            T_relation_embedding = self.least_square(tail_entity_pair_embeddings, tail_pair_embeddings)
            Hp_relation_embedding = H_relation_embedding
            Tp_relation_embedding = T_relation_embedding
            
            H_entity_embeddings = self.reverse_least_square(H_relation_embedding, H)
            T_entity_embeddings = self.reverse_least_square(T_relation_embedding, T)
            Hp_entity_embeddings = self.reverse_least_square(H_relation_embedding, Hp)
            Tp_entity_embeddings = self.reverse_least_square(T_relation_embedding, Tp)

            self.data_preprocessing.dic_relation_embeddings[r]['H'] = H_relation_embedding
            self.data_preprocessing.dic_relation_embeddings[r]['T'] = T_relation_embedding
            self.data_preprocessing.dic_relation_embeddings[r]['Hp'] = Hp_relation_embedding
            self.data_preprocessing.dic_relation_embeddings[r]['Tp'] = Tp_relation_embedding

            relation_entity_embeddings[r] = (iH, iT, iHp, iTp, H_entity_embeddings, T_entity_embeddings, Hp_entity_embeddings, Tp_entity_embeddings)

        self.set_entity_embeddings(relation_entity_embeddings)

        pos_corrs, neg_corrs = [], []
        for r in self.data_preprocessing.dic_positive_embedding.keys():
                
            iHT =  self.data_preprocessing.dic_positive_samples[r]
            iHpTp =  self.data_preprocessing.dic_negative_samples[r]
            
            iH, iT = iHT[:,0], iHT[:,1]
            iHp, iTp = iHpTp[:,0], iHpTp[:,1]

            H_entity_embeddings = self.data_preprocessing.entity_embeddings[iH]
            T_entity_embeddings = self.data_preprocessing.entity_embeddings[iT]
            Hp_entity_embeddings = self.data_preprocessing.entity_embeddings[iHp]
            Tp_entity_embeddings = self.data_preprocessing.entity_embeddings[iTp]

            pos_entity_pair_embeddings = np.c_[H_entity_embeddings, T_entity_embeddings]
            neg_entity_pair_embeddings = np.c_[Hp_entity_embeddings, Tp_entity_embeddings]
                
            H_relation_embedding = self.data_preprocessing.dic_relation_embeddings[r]['H']
            T_relation_embedding = self.data_preprocessing.dic_relation_embeddings[r]['T']
            Hp_relation_embedding = self.data_preprocessing.dic_relation_embeddings[r]['Hp']
            Tp_relation_embedding = self.data_preprocessing.dic_relation_embeddings[r]['Tp']

            H = self.map_entity_embedding_to_pair_embeddings(H_entity_embeddings, H_relation_embedding)
            T = self.map_entity_embedding_to_pair_embeddings(T_entity_embeddings, T_relation_embedding)
            Hp = self.map_entity_embedding_to_pair_embeddings(Hp_entity_embeddings, Hp_relation_embedding)
            Tp = self.map_entity_embedding_to_pair_embeddings(Tp_entity_embeddings, Tp_relation_embedding)

            self.data_preprocessing.dic_positive_embedding[r] = (H, T)
            self.data_preprocessing.dic_negative_embedding[r] = (Hp, Tp)

            Wh, Wt = self.mapping_to_correlation_space_matrices[r]

            LT = np.dot(T, Wt)
            LH = np.dot(H, Wh)
            LHp = np.dot(Hp, Wh)
            LTp = np.dot(Tp, Wt)

            pos_corr = np.mean(self.correlation(LH, LT))
            neg_corr = np.mean(self.correlation(LHp, LTp))

            pos_corrs.append(pos_corr)
            neg_corrs.append(neg_corr)

        mean_pos_corr = np.mean(pos_corrs)
        mean_neg_corr = np.mean(neg_corrs)
        print(f'r={r}, pos_corr={mean_pos_corr}, neg_corr={mean_neg_corr}, diff={mean_pos_corr - mean_neg_corr}')

        self.save(self.save_model_path)

    def mean_embedings(self, r, H, T, Hp, Tp):
        entity_count = len(self.data_structure.entities)
        pos_samples =  self.data_preprocessing.dic_positive_samples[r]
        neg_samples =  self.data_preprocessing.dic_negative_samples[r]
        iH, iT = pos_samples[:,0], pos_samples[:,1]
        iHp, iTp =  neg_samples[:,0], neg_samples[:,1]
        for i in range(entity_count):
            head_embeddings = np.concatenate([H[iH == i], Hp[iHp == i]])
            tail_embeddings = np.concatenate([T[iT == i], Tp[iTp == i]])
            if len(head_embeddings) > 0:
                head_embeddings = np.mean(head_embeddings, axis=0)
                H[iH == i] = head_embeddings
                Hp[iHp == i] = head_embeddings
            if len(tail_embeddings) > 0:
                tail_embeddings = np.mean(tail_embeddings, axis=0)
                T[iT == i] = tail_embeddings
                Tp[iTp == i] = tail_embeddings
        return H, T, Hp, Tp

    def least_square(self, entity_pair_embeddings, pair_embeddings):
        E = entity_pair_embeddings
        A = pair_embeddings
        R = np.linalg.pinv(E.T @ E) @ E.T @ A
        return R
    
    def reverse_least_square(self, relation_embedding, pair_embeddings):
        R = relation_embedding
        A = pair_embeddings
        E = A @ R.T @ np.linalg.pinv(R @ R.T)
        return E
    
    def set_entity_embeddings(self, relation_entity_embeddings):

        iH, iT, iHp, iTp, H_entity_embeddings, T_entity_embeddings, Hp_entity_embeddings, Tp_entity_embeddings = [], [], [], [], [], [], [], []
        for r, v in relation_entity_embeddings.items():
            r_iH, r_iT, r_iHp, r_iTp, r_H_entity_embeddings, r_T_entity_embeddings, r_Hp_entity_embeddings, r_Tp_entity_embeddings = v
            iH.append(r_iH)
            iT.append(r_iT)
            iHp.append(r_iHp)
            iTp.append(r_iTp)
            H_entity_embeddings.append(r_H_entity_embeddings)
            T_entity_embeddings.append(r_T_entity_embeddings)
            Hp_entity_embeddings.append(r_Hp_entity_embeddings)
            Tp_entity_embeddings.append(r_Tp_entity_embeddings)

        iH = np.concatenate(iH)
        iT = np.concatenate(iT)
        iHp = np.concatenate(iHp)
        iTp = np.concatenate(iTp)
        H_entity_embeddings = np.concatenate(H_entity_embeddings)
        T_entity_embeddings = np.concatenate(T_entity_embeddings)
        Hp_entity_embeddings = np.concatenate(Hp_entity_embeddings)
        Tp_entity_embeddings = np.concatenate(Tp_entity_embeddings)

        indices = np.r_[iH, iT, iHp, iTp]
        unique_indices = np.unique(indices)
        dim = self.latent_dimension

        for e in unique_indices:
            embeddings =  np.r_[
                H_entity_embeddings[iH == e, :dim],
                T_entity_embeddings[iT == e, :dim],
                Hp_entity_embeddings[iHp == e, :dim],
                Tp_entity_embeddings[iTp == e, :dim],
                H_entity_embeddings[iH == e, :dim],
                T_entity_embeddings[iT == e, :dim]
            ]
            entity_embedding = np.mean(embeddings, axis=0)
            self.data_preprocessing.entity_embeddings[e] = entity_embedding

        H_entity_embeddings = self.data_preprocessing.entity_embeddings[iH]
        T_entity_embeddings = self.data_preprocessing.entity_embeddings[iT]
        Hp_entity_embeddings = self.data_preprocessing.entity_embeddings[iHp]
        Tp_entity_embeddings = self.data_preprocessing.entity_embeddings[iTp]

        return H_entity_embeddings, T_entity_embeddings, Hp_entity_embeddings, Tp_entity_embeddings
    
    def map_entity_embedding_to_pair_embeddings(self, entity_pair_embeddings, relation_embedding):
        E = entity_pair_embeddings
        R = relation_embedding
        A = E @ R
        return A

    def predict(self):
        pass

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
                self.mapping_to_correlation_space_matrices = data.mapping_to_correlation_space_matrices
    
    def solve_tranfser_matrices_cca(self, H, T, Hp, Tp, dim):

        n , m = H.shape
        n_p, m = Hp.shape

        sigma_hh = np.dot(np.transpose(H), H) / (n - 1)
        sigma_tt = np.dot(np.transpose(T), T) / (n - 1)
        sigma_ht = np.dot(np.transpose(H), T) / (n - 1)
        sigma_hptp = np.dot(np.transpose(Hp), Tp) / (n_p - 1)

        A_left = np.dot(np.linalg.pinv(sigma_tt), sigma_ht.T-sigma_hptp.T)
        A_right = np.dot(np.linalg.pinv(sigma_hh), sigma_ht-sigma_hptp)
        A = np.dot(A_left, A_right)
        w, v = np.linalg.eig(A)
        
        w[w<0] = 0
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:,idx]

        ro = np.sqrt(w)
        Wt, ro = v[:, w > 0], ro[w > 0]
        Wh = np.dot(np.dot(np.linalg.pinv(sigma_hh), sigma_ht - sigma_hptp), Wt)
        Wh /= ro[np.newaxis,:]

        if Wh.shape[1] > dim:
            Wh = Wh[:,:dim]
            Wt = Wt[:,:dim]

        Wh = np.real(Wh)
        Wt = np.real(Wt)

        return Wh, Wt

    def solve_tranfser_matrices(self, H, T, Hp, Tp):

        n , m = H.shape
        n_p, m = Hp.shape

        sigma_hh = np.dot(np.transpose(H), H)
        sigma_tt = np.dot(np.transpose(T), T)
        sigma_ht = np.dot(np.transpose(H), T)
        sigma_hptp = np.dot(np.transpose(Hp), Tp)

        res1 = (1/n)*sigma_ht.T-(1/n_p)*sigma_hptp.T
        res2 = np.dot(res1, np.linalg.pinv(sigma_hh))
        res3 = np.dot(res2, (1/n)*sigma_ht-(1/n_p)*sigma_hptp)
        X = np.dot(np.linalg.pinv(sigma_tt), res3)
 
        w, v = np.linalg.eig(X)
        sum_lambda_2_lambda_4 = np.sqrt(w)

        w[np.isnan(w)] = 0
        idx = np.argsort(w)[-1]

        Wt, sum_lambda_1_lambda_3 = v, sum_lambda_2_lambda_4

        res1 = np.linalg.pinv(sigma_hh)
        res2 = np.dot(res1, (1/n)*sigma_ht-(1/n_p)*sigma_hptp)
        Wh = np.dot(res2, Wt)
        Wh = Wh * (1/(sum_lambda_1_lambda_3[np.newaxis,:]))

        return Wh, Wt

    def solve_embeddings(self, Wh, Wt, H, T, Hp, Tp):
        
        n , _ = H.shape
        n_p, _ = Hp.shape

        new_H = self.get_embedding(Wh, Wt, H, T, n)
        new_T = self.get_embedding(Wt, Wh, T, new_H, n)
        new_Hp = self.get_embedding(Wh, Wt, Hp, -Tp, n_p)
        new_Tp = self.get_embedding(Wt, Wh, Tp, -new_Hp, n_p)

        return new_H, new_T, new_Hp, new_Tp

    def get_embedding(self, Wh, Wt, H, T, n):

        c1 = np.linalg.pinv(np.dot(Wh, Wh.T))
        c2 = np.dot(Wt, Wh.T)

        res1 = (1/(n**2))*np.linalg.multi_dot([Wh.T, c1, c2.T, T.T, T, c2, c1, Wh])
        res1[res1 < 0] = 0
        res2 = np.sqrt(np.diag(res1))
        res3 = np.diag(res2)

        c3 = np.linalg.pinv(np.linalg.multi_dot([Wh, res3, Wh.T]))

        new_H = np.linalg.multi_dot([T, c2, c3*(1/n)])

        return new_H

    def normalize(self, H, T, Hp, Tp):

        H = np.real(H)
        T = np.real(T)
        Hp = np.real(Hp)
        Tp = np.real(Tp)
        
        H = normalize(H, axis=1, norm='l2')
        T = normalize(T, axis=1, norm='l2')
        Hp = normalize(Hp, axis=1, norm='l2')
        Tp = normalize(Tp, axis=1, norm='l2')

        return H, T, Hp, Tp