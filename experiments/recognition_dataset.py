import numpy as np
import os
from copy import deepcopy
from collections import Counter
import time


class RecognitionTestor:
    
    algorithms = ["facenet", "mobilefacenet", "vggface", "arcface", "cosface"]
    
    tolerances = {
            "cosface" : 1.24,
            "vggface" : 185.0,
            "facenet" : 1.0,
            "arcface" : 0.6,
            "mobilefacenet" : 1.25,
        }
    
    times = {
            "cosface" : 0,
            "vggface" : 0,
            "facenet" : 0,
            "arcface" : 0,
            "mobilefacenet" : 0,
        }
    
    
    def __init__(self, folderpath):
        
        self.folderpath = folderpath
        self.n_parts = len(os.listdir(folderpath))/3
        self.embeddings_all, self.dataset_all, self.people_all = self.load_embeddings(folderpath)
        
    
    def set_size(self, n):
        
        while(True):
            
            permutations = np.random.permutation(len(self.people_all))
            self.people = self.people_all[permutations[:n]]
            filter_func = np.vectorize(lambda x: x in self.people)
            self.perm = filter_func(self.dataset_all[:, 0])
            self.dataset = self.dataset_all[self.perm]
            embeddings = deepcopy(self.embeddings_all)
            for key in embeddings:
                embeddings[key] = embeddings[key][self.perm]
            self.embeddings = embeddings
            
            counter = Counter(self.dataset[:,0]).most_common()
            self.people_with_multiple = list(filter(lambda x: x[1] >1, counter))
            self.people_with_multiple = list(map(lambda x: x[0], self.people_with_multiple))
            max_n = counter[0][1]
            if(max_n >= 2):
                break

    def load_embeddings(self, folderpath):
    
        embeddings = {
                "cosface" : [],
                "vggface" : [],
                "facenet" : [],
                "arcface" : [],
                "mobilefacenet" : [],
            }    
        
        iter_i = 1
        dataset = []
        people = []
        
        while( os.path.exists(os.path.join(folderpath, f"embeddings_{iter_i}.npy")) ):
            
            try:
                embedding_array = np.load(os.path.join(folderpath, f"embeddings_{iter_i}.npy"), allow_pickle = True)
                dataset_array = np.load(os.path.join(folderpath, f"dataset_{iter_i}.npy"), allow_pickle = True)
                people_array = np.load(os.path.join(folderpath, f"people_{iter_i}.npy"), allow_pickle = True)
            
                dataset.append(dataset_array)
                people.append(people_array.reshape(-1, 1))
                for i in range(5):
                    embeddings[embedding_array[i, 0]].append(embedding_array[i, 1])
                
                iter_i = iter_i + 1
                if(iter_i>10): break
            except Exception as e:
                #print(e)
                if(iter_i < self.n_parts):
                    iter_i = iter_i + 1
                else:
                    break
     
        dataset = np.vstack(dataset)
        people = np.vstack(people)
        for key in embeddings:
            embeddings[key] = np.vstack(embeddings[key])
            
        return embeddings, dataset, people
    
    
    def run_experiment(self, n):
        
        for i in range(n):
            pass
        
        
    def calculate_face_distance(self, face, face_encodings):
    
        if len(face_encodings) == 0:
            return np.empty((0))
    
        distances = np.linalg.norm(face_encodings - face, axis=1)
        
        return distances
    
    
    def run_test(self, keep_person = True):
        
        embeddings = deepcopy(self.embeddings)  
        dataset = deepcopy(self.dataset)
        chose_from_collection = self.people_with_multiple if(keep_person) else self.people.ravel()
        person = np.random.choice(chose_from_collection)
        
        inds = np.argwhere(dataset[:, 0] == person).ravel()
        row_i = np.random.choice(inds)
        name, _, filepath, age, gender, race = dataset[row_i]
        n_occurences = len(inds) - 1
        person_info = [age, gender, race, n_occurences]
        
        selected_embeddings = {}
        for algorithm in embeddings:
            selected_embeddings[algorithm] = embeddings[algorithm][row_i].reshape(1, -1)
        
        inds_to_delete = [row_i] if keep_person else inds
        dataset = np.delete(dataset, inds_to_delete, 0)
        assert not np.any(dataset[:, 2] == filepath)
        correct_inds = np.argwhere(dataset[:, 0] == person).ravel()
        for key in embeddings:
            embeddings[key] = np.delete(embeddings[key], inds_to_delete, 0)
            
            
            
        return person, person_info, selected_embeddings, correct_inds, dataset, embeddings
        
        
        
    
    
    


t = RecognitionTestor("embeddings/lfw")
t.set_size(50)

keep_person = False
person, person_info, selected_embeddings, correct_inds, dataset, embeddings = t.run_test(keep_person)

assert np.all(np.argwhere(dataset[:, 0] == person).ravel() == correct_inds)
assert np.all([embeddings[k].shape[0] == dataset.shape[0] for k in embeddings])

for algorithm in t.algorithms:
    
    tolerance = round( t.tolerances[algorithm]  , 2)
    all_embeddings = embeddings[algorithm] 
    selected_vector = selected_embeddings[algorithm]
    distances = t.calculate_face_distance(selected_vector, all_embeddings)
    similarity_order = np.argsort(distances)
    best_match_ind = similarity_order[0]
    best_match_distance = distances[best_match_ind]
    recognized = best_match_distance < tolerance
    is_correct = recognized == keep_person and (not keep_person or (best_match_ind in correct_inds))
    
    print(algorithm[:7],  (is_correct, recognized, tolerance, round(best_match_distance, 2),  ),  (list(similarity_order[:3]), list(correct_inds)) )
    




















