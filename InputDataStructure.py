import numpy as np
import os
import re
import pickle as pk

class InputDataStructure:

    def __init__(self, triples, save_data_path):
        
        if not self.load(save_data_path):
            entities, relations = [], []

            for h,r,t in triples:

                entities.extend([h, t])
                relations.append(r)

            entities, relations = list(np.unique(entities)), list(np.unique(relations))

            dic_id_entity   = { id:str(entity) for id, entity in enumerate(entities) }
            dic_id_relation = { id:str(rel)    for id, rel    in enumerate(relations) }
            dic_entity_id   = { str(entity):id for id, entity in enumerate(entities) }
            dic_relation_id = { str(rel):id    for id, rel    in enumerate(relations) }

            triples_id = [(dic_entity_id[h], dic_relation_id[r], dic_entity_id[t]) for h,r,t in triples]

            self.triples, self.triples_id = triples, triples_id
            self.entities, self.relations = entities, relations
            self.dic_id_entity, self.dic_id_relation = dic_id_entity, dic_id_relation
            self.dic_entity_id, self.dic_relation_id = dic_entity_id, dic_relation_id

            self.save(save_data_path)
       
    def save(self, data_path):
        pass
        #with open(data_path, 'wb') as f:
        #    pk.dump(self, f, pk.HIGHEST_PROTOCOL)
            
    def load(self, data_path):
        
        if not os.path.isfile(data_path):
            return False
        else:
            with open(data_path, 'rb') as f:
                data = pk.load(f)
                
                self.triples, self.triples_id = data.triples, data.triples_id
                self.entities, self.relations = data.entities, data.relations
                self.dic_id_entity, self.dic_id_relation = data.dic_id_entity, data.dic_id_relation
                self.dic_entity_id, self.dic_relation_id = data.dic_entity_id, data.dic_relation_id

                return True
