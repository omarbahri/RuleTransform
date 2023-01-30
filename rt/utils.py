import numpy as np
from sklearn.base import clone
import pandas as pd
from pathlib import Path
import pickle
import random

from shapelets import ContractedShapeletTransform

#Fit the transformer to all dimensions of the dataset
class MultivariateTransformer:
    def __init__(self, st):
        self.st = st
        self.sts = None
        
    def fit(self, X, y=None):
        self.n_dims = X.shape[1]
        self.sts = [clone(self.st) for _ in range(self.n_dims)]
        
        for i, transformer in enumerate(self.sts):
            transformer.fit(X.iloc[:,i].to_frame(), y)
        return self
    
    def transform(self, X, y=None):
        X_transformed = []
        for i, transformer in enumerate(self.sts):
            try:
                X_transformed.append(transformer.transform(X.iloc[:,i].to_frame()))
            except RuntimeError:
                continue
                                 
        X_new = pd.concat(X_transformed, axis=1)
        return X_new

#write transformer to file
def save_transformer(parent_dir, name, transformer):
    Path(parent_dir + "/" + name).mkdir(parents=True, exist_ok=True)
    
    with open(parent_dir + "/" + name + "/" + name +  "_shapelets.pkl", 'wb') as f:
            pickle.dump(get_shapelets(transformer), f)
    # np.save(parent_dir + "/" + name + "/" + name +  "_indices.npy", get_indices(transformer))
    # np.save(parent_dir + "/" + name + "/" + name +  "_scores.npy", get_scores(transformer))
        
#save shapelets distances only (for test set)
def save_shapelets_distances(parent_dir, name, transformer, test=False):
    Path(parent_dir + "/" + name).mkdir(parents=True, exist_ok=True)
    
    if test:
        #np.save(parent_dir + "/" + name + "/" + name +  "_shapelets_distances_test.npy", get_shapelets_distances(transformer))
        with open(parent_dir + "/" + name + "/" + name +  "_shapelets_distances_test.pkl", 'wb') as f:
            pickle.dump(get_shapelets_distances(transformer), f)
    else:
        #np.save(parent_dir + "/" + name + "/" + name +  "_shapelets_distances.npy", get_shapelets_distances(transformer))
        with open(parent_dir + "/" + name + "/" + name +  "_shapelets_distances.pkl", 'wb') as f:
            pickle.dump(get_shapelets_distances(transformer), f)
    
#get the list of shapelets of a transformer
def get_shapelets(transformer):    
    all_shapelets = []
    
    for st in transformer.sts:
        dim_shapelets = []
        for shapelet in st.shapelets:
            dim_shapelets.append(shapelet.data)
        all_shapelets.append(dim_shapelets)
        
    return all_shapelets

#get the list of shapelet indices of a transformer
def get_indices(transformer):    
    all_indices = []
    
    for st in transformer.sts:
        dim_indices = []
        for shapelet in st.shapelets:
            ind = np.empty(3, dtype=np.uint16)
            ind[0] = shapelet.series_id
            ind[1] = shapelet.start_pos
            ind[2] = shapelet.start_pos + shapelet.length
            dim_indices.append(ind)
        all_indices.append(dim_indices)
        
    return np.asarray(np.asarray(all_indices))
    
#get the list of shapelet scores of a transformer
def get_scores(transformer):
    all_scores = []
    
    for st in transformer.sts:
        dim_scores = []
        for shapelet in st.shapelets:
            dim_scores.append(shapelet.info_gain)
        all_scores.append(dim_scores)
        
    return np.asarray(np.asarray(all_scores))
    
#get the distance of shapelets from each other shapelet in the MTS
def get_shapelets_distances(transformer):
    all_shapelets_distances = []
    
    for st in transformer.sts:
        shapelets_distances = []
        for shapelet in st.shapelets:
            shapelets_distances.append(shapelet.distances)
        
        all_shapelets_distances.append(shapelets_distances)
    return all_shapelets_distances

#Find the interval of shapelets lengths to mine from the dataset
def get_shapelets_lengths_interval(X, y, total_time, random_state=42):
    max_length = X.iloc[0][0].shape[0]
    min_length = 3
    
    n_samples = X.shape[0]
    n_dims = X.shape[1]
    
    random_state = 10
    
    time_contract_per_dimension = total_time/10
    
    #List to hold the lengths of all mined shapelets
    all_lengths = []
        
    for i in range(10):
        random.seed(random_state)
        ids = [random.randint(0, n_samples-1) for p in range(0, 10)]
        d = random.randint(0,n_dims-1)
        
        sub_X = X.iloc[ids, d].to_frame()
        sub_y = y[ids]
        
        
        # Shapelet transformation
        st = ContractedShapeletTransform(
            time_contract_in_mins=time_contract_per_dimension,
            num_candidates_to_sample_per_case=10,
            min_shapelet_length=min_length,
            max_shapelet_length=max_length,
            predefined_ig_rejection_level=0.005,
            verbose=0,
        )
                
        st.fit(sub_X, sub_y)
        
        #lengths = []
                
        for shapelet in st.shapelets[:20]:
            all_lengths.append(shapelet.length)

        random_state += 1
        
    #We need to extract 100 shapelets 
    while len(all_lengths)<100:
         ids = [random.randint(0, n_samples-1) for p in range(0, 10)]
         d = random.randint(0,n_dims-1)
         
         sub_X = X.iloc[ids, d].to_frame()
         sub_y = y[ids]
         
         # Shapelet transformation
         st = ContractedShapeletTransform(
             time_contract_in_mins=time_contract_per_dimension,
             num_candidates_to_sample_per_case=10,
             min_shapelet_length=min_length,
             max_shapelet_length=max_length,
             verbose=0,
         )
                 
         st.fit(sub_X, sub_y)
         
                 
         for shapelet in st.shapelets:
             all_lengths.append(shapelet.length)
                 
         random_state += 1
        
    #If we collected more than 100 shapelets, select 100 random ones
    all_lengths = random.sample(all_lengths, 100)
        
    all_lengths.sort()
            
    return all_lengths[24], all_lengths[74]

def multivariate_contracted_st(X_train, y_train, X_test, time_contract, lengths_contract):       
    #Get lengths of shapelets to mine
    min_length, max_length = get_shapelets_lengths_interval(X_train, y_train, total_time=lengths_contract)
    
    
    # How long (in minutes) to extract shapelets for.
    # This is a simple lower-bound initially;
    # once time is up, no further shapelets will be assessed
    time_contract_in_mins = time_contract
    
    time_contract_in_mins_per_dim = int(time_contract_in_mins/X_train.shape[1])
    
    #If the time contract per dimensions is less than one minute, sample 
    #time_contract_in_mins random dimensions and apply the ST to them
    seed = 10
    
    if time_contract_in_mins_per_dim < 1:
        random.seed(seed)
        dims = [random.randint(0, X_train.shape[1]-1) for p in range(0, int(time_contract_in_mins))]
            
        X_train = X_train.iloc[:, dims]
        
        #Spend one minute on each dimension
        time_contract_in_mins_per_dim = 1
    
    # The initial number of shapelet candidates to assess per training series.
    # If all series are visited and time remains on the contract then another
    # pass of the data will occur
    initial_num_shapelets_per_case = 10
    
    # Whether or not to print on-going information about shapelet extraction.
    # Useful for demo/debugging
    verbose = 2
    
    st = ContractedShapeletTransform(
        time_contract_in_mins=time_contract_in_mins_per_dim,
        num_candidates_to_sample_per_case=initial_num_shapelets_per_case,
        min_shapelet_length=min_length,
        max_shapelet_length=max_length,
        verbose=verbose,
        predefined_ig_rejection_level=0.001,
    )
    
    transformer = MultivariateTransformer(st)
    
    transformer.fit(X_train, y_train)
    
    X_new = transformer.transform(X_train)    
    
    X_test_new = transformer.transform(X_test)

    return X_new, X_test_new
    

