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
            "cosface" : 0.974,
            "vggface" : 0.667,
            "facenet" : 0.602,
            "arcface" : 1.594,
            "mobilefacenet" : 0.510,
        }
    
    
    def __init__(self, folderpath):
        
        self.folderpath = folderpath
        self.dataset_name = folderpath.split("/")[1]
        self.n_parts = len(os.listdir(folderpath))/3
        self.embeddings_all, self.dataset_all, self.people_all = self.load_embeddings(folderpath)
        
        self.n_people_all = np.unique(self.people_all).shape[0] 
        self.n_images_all = self.dataset_all.shape[0] 

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
        
        all_files = list(os.listdir(folderpath))
        embeddings_files = list(filter(lambda filename: "embeddings" in filename, all_files))
        inds = list(set(map( lambda filename: filename.split("_")[1].split(".")[0], embeddings_files )))
        #print(inds)
        for iter_i in inds:
            
            try:
                
                embedding_array = np.load(os.path.join(folderpath, f"embeddings_{iter_i}.npy"), allow_pickle = True)
                dataset_array = np.load(os.path.join(folderpath, f"dataset_{iter_i}.npy"), allow_pickle = True)
                people_array = np.load(os.path.join(folderpath, f"people_{iter_i}.npy"), allow_pickle = True)
                #print(f"{iter_i}")
                vector_checksum = dataset_array.shape[0] == np.array([embedding_array[i, 1].shape[0] for i in range(5)])
                assert np.all(vector_checksum)
                dataset.append(dataset_array)
                people.append(people_array.reshape(-1, 1))
                for i in range(5):
                    embeddings[embedding_array[i, 0]].append(embedding_array[i, 1])

            except Exception as e:
                #print(e)
                pass
     
        dataset = np.vstack(dataset)
        people = np.vstack(people)
        for key in embeddings:
            embeddings[key] = np.vstack(embeddings[key])
            
            
        return embeddings, dataset, people
    
    
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
            
        self.n_people = np.unique(self.people).ravel().shape[0]
        self.n_images = self.dataset.shape[0]
        
        
        
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
        assert len(inds) > 0
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
                        
        assert np.all(np.argwhere(dataset[:, 0] == person).ravel() == correct_inds)
        assert np.all([embeddings[k].shape[0] == dataset.shape[0] for k in embeddings])
        
        rows = []
        
        for algorithm in self.algorithms:
            
            start_time = time.time()
            tolerance = self.tolerances[algorithm]
            all_embeddings = embeddings[algorithm] 
            selected_vector = selected_embeddings[algorithm]
            distances = self.calculate_face_distance(selected_vector, all_embeddings)
            similarity_order = np.argsort(distances)
            best_match_ind = similarity_order[0]
            best_match_distance = distances[best_match_ind]
            recognized = best_match_distance < tolerance
            is_correct = recognized == keep_person and (not keep_person or (best_match_ind in correct_inds))
            
            if(keep_person and len(correct_inds) > 0):
                ages = dataset[correct_inds, 3].astype(int)
                selected_age = int(person_info[0])
                min_age_difference_ind = np.argmin(np.absolute( ages - selected_age ))
                min_age_difference = np.absolute(ages[min_age_difference_ind] - selected_age)
            else:
                min_age_difference = None
                
                
            n_occurences = len(correct_inds)
            age, gender, race, _ = person_info
            match_name, _, _, match_age, match_gender, match_race = dataset[best_match_ind]
            
            comparison_time = (time.time() - start_time)*3.5
            time_taken = self.times[algorithm] + comparison_time
            
            row = [
                    self.dataset_name, self.n_people, self.n_images, algorithm, person, keep_person,
                    best_match_distance,  tolerance, recognized, is_correct, time_taken, comparison_time, age, gender, race,
                    min_age_difference, match_name, match_age, match_gender, match_race,
                   ]
            row = [str(item) for item in row]
            rows.append(row)
            
        return rows
            




    def run_experiment(self, n):
        
        for i in range(n):
            pass
    



results_file = "results/results_recognition.csv"


columns = [
     "dataset", "n_people", "n_images", "algorithm", "person", "in_dataset", "min_distance" , 
     "tolerance", "recognized", "correct", "total_time", "comparison_time", "age", "gender", "race",
     "min_age_difference", "match_name", "match_age", "match_gender", "match_race",
    ]

if(not os.path.exists(results_file)):
    with open(results_file, "w") as f:
        column_line = ";".join(columns) + "\n"
        f.writelines([column_line])


recognitions_per_size = 3072
batch_size = 512


testors = [
            RecognitionTestor("embeddings/lfw"),    
            RecognitionTestor("embeddings/cacd"),    
          ]

print([t.n_images_all for t in testors])
print([t.n_people_all for t in testors])


for testor in testors:
    
    print(f"\nTesting: {testor.dataset_name}")
    for N in [10, 25, 50, 100, 200, 500, 1000, 2000]:
        
        N = min(N, testor.n_people_all)
        
        print(f"Performing tests with N = {N}")
        testor.set_size(N)
        
        for _ in range(int(recognitions_per_size / batch_size)):
            
            all_rows = []
            
            for _ in range(batch_size):
                keep_person = np.random.random() <= 0.5
                try:
                    rows = testor.run_test(keep_person)
                except:
                    continue
                all_rows = all_rows + rows
                
                
            func = lambda row: ";".join(row) + "\n"
            rows_str = list(map(func, all_rows))
            
            with open(results_file, "a") as f:
                f.writelines(rows_str)
                
                
        if(N >= testor.n_people_all):
            break
            
    













