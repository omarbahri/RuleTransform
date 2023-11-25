#Author: Omar Bahri

from sklearn.base import clone
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import random
import bottleneck
from numba import jit, njit, prange
import time

from tslearn.clustering import KernelKMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import dtw


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

#Calculate the interval of shapelets lengths to mine for the dataset
def get_shapelets_lengths_interval(X, y, total_time):
    max_length = X.iloc[0][0].shape[0]
    min_length = 3
    
    n_samples = X.shape[0]
    n_dims = X.shape[1]
    
    seed = 10
    
    time_contract_per_dimension = total_time/10
    
    #List to hold the lengths of all mined shapelets
    all_lengths = []
        
    for i in range(10):
        print(len(all_lengths))
        random.seed(seed)
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

        seed += 1
        
    #We need to extract 100 shapelets 
    while len(all_lengths)<100:
         print(len(all_lengths))
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
                 
         seed += 1
        
    #If we collected more than 100 shapelets, select 100 random ones
    all_lengths = random.sample(all_lengths, 100)
        
    all_lengths.sort()
            
    return all_lengths[24], all_lengths[74]

#Calculate the interval of shapelets lengths to mine for the dataset (returns a discrete
#set of lengths)
def get_shapelets_lengths_interval_discrete(X, y, n_lengths, total_time):
    max_length = X.iloc[0][0].shape[0]
    min_length = 3
    
    n_samples = X.shape[0]
    n_dims = X.shape[1]
    
    seed = 10
    
    time_contract_per_dimension = total_time/10
    
    #List to hold the lengths of all mined shapelets
    all_lengths = []
        
    for i in range(10):
        print(len(all_lengths))
        random.seed(seed)
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
        
                
        for shapelet in st.shapelets[:20]:
            all_lengths.append(shapelet.length)
                
        seed += 1
           
    #We need to extract 100 shapelets 
    while len(all_lengths)<100:
        print(len(all_lengths))
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
                
        seed += 1
        
    #If we collected more than 100 shapelets, select 100 random ones
    all_lengths = random.sample(all_lengths, 100)
        
    all_lengths.sort()
    
    step = int(50/(n_lengths-1)) + 1
    
    lengths = []
    
    j = 24
        
    for i in range(n_lengths-1):
        lengths.append(all_lengths[j+i*step])
        
    lengths.append(all_lengths[74])
    
    return lengths

#Given the shapelet_locations and shapelet_distances of one single shapelet, removes
#the overlapping shapelet locations except the closest to the original
def remove_similar_locations(shapelet_locations, shapelet_distances):
    #List to keep indices to be discarded
    to_discard = []
    
    #Sort the shapelet_locations by sample index, then by start index
    shapelet_locations = shapelet_locations[np.lexsort((shapelet_locations[:,1],shapelet_locations[:,0]))]

    #Variables to store the currently selected shapelet
    current_dist = np.inf
    current_idx = -1    #the sample index
    current_start = -1
    current_end = -1
    current_i = -1  #the index in the shapelet_locations array
    
    i=0
    for shapelet in shapelet_locations:
        idx = shapelet[0]
        start = shapelet[1]
        end = shapelet[2]
        #Check if this location overlaps the selected shapelet
        if (idx == current_idx and (not (start>=current_end or end<=current_start))):
            dist = shapelet_distances[idx][start]

            #If the distance of this shapelet is smaller than the distance of the currently
            #selected shapelet, discard the currently selected shapelet and select this one
            if (dist < current_dist):
                to_discard.append(current_i)
                current_i = i
                current_dist = dist
                #Widen shapelet l
                current_start = np.minimum(current_start, start)
                current_end = np.maximum(current_end, end)
            #Else, discard this shapelet               
            else:   
                to_discard.append(i)
        #If it doesn't overlap it, select this one     
        else:
            current_idx = shapelet[0]
            current_start = shapelet[1]
            current_end = shapelet[2]
            current_dist = shapelet_distances[idx][start]
            current_i = i
        i+=1
    
    return np.delete(shapelet_locations, to_discard, axis=0)

#Given the shapelet_distances matrix of a given shapelet, get the locations of 
#the closest shapelets from the entire dataset based on a chosen percentage
def get_shapelet_locations(shapelet_distances, ts_length, percentage):
    #n is the number of similar shapelets to select
    n = int(percentage*shapelet_distances.shape[0]*shapelet_distances.shape[1])
    
    #Get the indices of the n closest shapelets to the original shapelet
    idx = bottleneck.argpartition(shapelet_distances.ravel(),n)[:n]
    s_indices = np.stack(np.unravel_index(idx, shapelet_distances.shape)).T
    
    #Compute the length of the shapelet
    shapelet_length = ts_length - shapelet_distances.shape[1] + 1
    
    #Create an array to store the locations of the closest n shapelets
    shapelet_locations = np.empty((s_indices.shape[0], s_indices.shape[1]+1), dtype=np.uint16)
    #Each shapelet is represented by (sample_index, start, end)
    for i in range(shapelet_locations.shape[0]):
        shapelet_locations[i] = np.append(s_indices[i], s_indices[i][1]+shapelet_length)
    
    #Remove overlapping shapelets and keep the closest one to th original shapelet
    shapelet_locations = remove_similar_locations(shapelet_locations, shapelet_distances)
    
    return shapelet_locations

#Get the locations of the closest shapelets for each shapelet across the 
#entire dataset based on a chosen percentage
def get_all_shapelet_locations(shapelets_distances, ts_length, percentage):
    all_shapelet_locations = []
    for dim in shapelets_distances:
        dim_shapelet_locations = []
        for shapelet in dim:
            dim_shapelet_locations.append(get_shapelet_locations(shapelet, ts_length, percentage))
        all_shapelet_locations.append(dim_shapelet_locations)
        
    return all_shapelet_locations

#Given the shapelet_distances matrix of a given shapelet, get the locations of 
#the closest shapelets from the entire dataset 
def get_shapelet_locations_scaled_threshold(shapelet_distances, ts_length, threshold):
    #Compute the length of the shapelet
    shapelet_length = ts_length - shapelet_distances.shape[1] + 1
    
    #Get the indices of the n closest shapelets to the original shapelet
    s_indices = []
    for i in range(shapelet_distances.shape[0]):
        for j in range(shapelet_distances.shape[1]):
            #Compare to the threshold, scaled to shapelet length
            if shapelet_distances[i][j]/shapelet_length <= threshold:
                # print(shapelet_distances[i][j]/shapelet_length,i)
                s_indices.append(np.array([i,j]))
       
    if len(s_indices)>0:
        s_indices = np.asarray(s_indices) 
        
        #Create an array to store the locations of the closest n shapelets
        shapelet_locations = np.empty((s_indices.shape[0], s_indices.shape[1]+1), dtype=np.uint32)
        #Each shapelet is represented by (sample_index, start, end)
        for i in range(shapelet_locations.shape[0]):
            shapelet_locations[i] = np.append(s_indices[i], s_indices[i][1]+shapelet_length)
        
        #Remove overlapping shapelets and keep the closest one to th original shapelet
        shapelet_locations = remove_similar_locations(shapelet_locations, shapelet_distances)
    
        return shapelet_locations
    
    else:
        return np.array(np.array([[-1,-1,-1]]), dtype=np.uint32)

#Get the locations of the closest shapelets for each shapelet across the 
#entire dataset based on a chosen percentage
def get_all_shapelet_locations_scaled_threshold(shapelets_distances, ts_length, percentage):
    #Get the threshold to be used for selecting shapelet occurences
    threshold = get_occurences_threshold(shapelets_distances, ts_length, percentage)
    
    all_shapelet_locations = []
    for dim_i, dim in enumerate(shapelets_distances):
        dim_shapelet_locations = []
        # print('DIM: ', dim_i)
        for shapelet in dim:
            dim_shapelet_locations.append(get_shapelet_locations_scaled_threshold(shapelet, ts_length, threshold))
        all_shapelet_locations.append(dim_shapelet_locations)
        
    return all_shapelet_locations, threshold

#Get the locations of the closest shapelets for each shapelet across the 
#entire dataset based on threshold from training set
def get_all_shapelet_locations_scaled_threshold_test(shapelets_distances, ts_length, threshold):    
    all_shapelet_locations = []
    for dim_i, dim in enumerate(shapelets_distances):
        dim_shapelet_locations = []
        # print('DIM: ', dim_i)
        for shapelet in dim:
            dim_shapelet_locations.append(get_shapelet_locations_scaled_threshold(shapelet, ts_length, threshold))
        all_shapelet_locations.append(dim_shapelet_locations)
    
    return all_shapelet_locations

#Given the cluster_distances matrix of a given cluster, get the locations of 
#the closest shapelets from the entire dataset 
def get_cluster_locations_scaled_threshold(cluster_distances, ts_length, threshold):
    #Get the arrays of distances without instance indices
    sds = np.asarray(list(zip(*cluster_distances))[1])
    
    #Get the arrays of instance indices only
    sds_idx = np.asarray(list(zip(*cluster_distances))[0])
    
    #Count the number of all possible occurences and get the maximum number of 
    #occurences per sample (corresponding to the shortest shapelet)
    n_occurences = 0
    max_occs = 0
    for sd in sds:
        occs = sd.shape[0]
        n_occurences += occs
        if occs > max_occs:
            max_occs = occs                          
    
    #Make arrays of occurences the same size for all shapelets (fill up the
    #missing ones from long shapelets with a large value), and move to new 
    #array (necessary in order to be able to cast to float)
    #Also, get each shapelet length and store in a list
    new_sds = np.full((sds.shape[0], max_occs), fill_value=np.inf, dtype=np.float64)
    shapelets_lengths = []
        
    #print(sds_idx)
    
    for i in range(len(sds)):   
        new_sds[i,0:sds[i].shape[0]] = sds[i]
        shapelets_lengths.append(ts_length - sds[i].shape[0] + 1)
        
    del sds
    
    #Get the indices of the n closest shapelets to the original shapelet
    s_indices = []
    for i in range(new_sds.shape[0]):
        for j in range(new_sds.shape[1]):
            #Compare to the threshold, scaled to shapelet length
            if new_sds[i][j]/shapelets_lengths[i] <= threshold:
                s_indices.append(np.array([i,j]))
    
    if len(s_indices)>0:
        s_indices = np.asarray(s_indices) 
        
        #Create an array to store the locations of the closest n shapelets
        shapelet_locations = np.empty((s_indices.shape[0], s_indices.shape[1]+1), dtype=np.uint32)
        #Each shapelet is represented by (sample_index, start, end)
        for i in range(shapelet_locations.shape[0]):
            #shapelet_locations[i] = np.append(s_indices[i], s_indices[i][1]+shapelets_lengths[i])
            shapelet_locations[i] = np.array([sds_idx[s_indices[i][0]], s_indices[i][1], s_indices[i][1]+shapelets_lengths[s_indices[i][0]]])        
    
        #Remove overlapping shapelets and keep the closest one to th original shapelet
        shapelet_locations = remove_similar_locations(shapelet_locations, new_sds)
    
        return shapelet_locations
    
    else:
        return np.array(np.array([[-1,-1,-1]]), dtype=np.uint32)
    
#Get the locations of the closest shapelets for each cluster-shapelet across the 
#entire dataset based on a chosen percentage
def get_all_cluster_locations_scaled_threshold(clusters_distances, ts_length, threshold):    
    all_cluster_locations = []
    for dim in clusters_distances:
        dim_cluster_locations = []
        for cluster in dim:
            dim_cluster_locations.append(get_cluster_locations_scaled_threshold(cluster, ts_length, threshold))
        all_cluster_locations.append(dim_cluster_locations)
        
    return all_cluster_locations

#Given the shapelet_distances matrix of a given cluster-shapelet, get the locations of 
#the closest shapelets from the entire dataset based on a chosen percentage
def get_cluster_locations(shapelet_distances, ts_length, percentage):
    #Get the arrays of distances without instance indices
    sds = np.asarray(list(zip(*shapelet_distances))[1])
    
    #Get the arrays of instance indices only
    sds_idx = np.asarray(list(zip(*shapelet_distances))[0])
    
    #Count the number of all possible occurences and get the maximum number of 
    #occurences per sample (corresponding to the shortest shapelet)
    n_occurences = 0
    max_occs = 0
    for sd in sds:
        occs = sd.shape[0]
        n_occurences += occs
        if occs > max_occs:
            max_occs = occs
    
    #n is the number of similar shapelets to select
    n = int(percentage*n_occurences)
    
    #Make arrays of occurences the same size for all shapelets (fill up the
    #missing ones from long shapelets with a large value), and move to new 
    #array (necessary in order to be able to cast to float)
    #Also, get each shapelet length and store in a list
    new_sds = np.full((sds.shape[0], max_occs), fill_value=np.inf, dtype=np.float64)
    shapelets_lenghts = []
    
    for i in range(len(sds)): 
        new_sds[i,0:sds[i].shape[0]] = sds[i]
        shapelets_lenghts.append(ts_length - sds[i].shape[0] + 1)
        
    del sds
    
    #Get the indices of the n closest shapelets to the original shapelet
    idx = bottleneck.argpartition(new_sds.ravel(),n)[:n]
    s_indices = np.stack(np.unravel_index(idx, new_sds.shape)).T
    
    #Create an array to store the locations of the closest n shapelets
    shapelet_locations = np.empty((s_indices.shape[0], s_indices.shape[1]+1), dtype=np.uint16)
    #Each shapelet is represented by (sample_index, start, end)
    for i in range(shapelet_locations.shape[0]):
        shapelet_locations[i] = np.array([sds_idx[s_indices[i][0]], s_indices[i][1], s_indices[i][1]+shapelets_lenghts[s_indices[i][0]]])
    
    #Remove overlapping shapelets and keep the closest one to th original shapelet
    shapelet_locations = remove_similar_locations(shapelet_locations, new_sds)
    
    return shapelet_locations

#Get the locations of the closest shapelets for each cluster-shapelet across the 
#entire dataset based on a chosen percentage
def get_all_cluster_locations(shapelets_distances, ts_length, percentage):
    all_shapelet_locations = []
    for dim in shapelets_distances:
        dim_shapelet_locations = []
        for shapelet in dim:
            dim_shapelet_locations.append(get_cluster_locations(shapelet, ts_length, percentage))
        all_shapelet_locations.append(dim_shapelet_locations)
        
    return all_shapelet_locations

#Cluster similar shapelets
#Return the indices of shapeletes contained in each cluster, at each dimension
def cluster_shapelets_kmeans(shapelets, n_clusters):  
    #List to keep track of the shapelets in each clusterm across all dimensions
    all_clusters_shapelets = []
        
    #Cluster each dimension
    for dim in range(len(shapelets)):
        #Format shapelets into dataset
        S = []
        
        for shapelet in shapelets[dim]:
            s=[]
            for e in shapelet[0]:   
                s.append(e)
            S.append(s) 
        
        S = to_time_series_dataset(S)
        
        n_c = n_clusters[dim]

        print("n_c: " + str(n_c))
        
        if n_c==1 or n_c==0:
            print("Target number of clusters is equal to " + str(n_c) + ". No clustering for this dimension=" + str(dim))
            all_clusters_shapelets.append(np.array([np.arange(0,len(shapelets[dim]))]))
            continue

        #Cluster and get the labels of shapelets in each cluster
        #gak_km = KernelKMeans(n_clusters=n_c, kernel="gak", random_state=0, verbose=1)

        model = TimeSeriesKMeans(n_clusters=n_c, metric="dtw", verbose=1,
                                 max_iter=10, random_state=0)
        #labels = gak_km.fit_predict(S)
        labels = model.fit_predict(S)
        
        print(len(labels))
        print("number of clusters found: " + str(np.unique(np.array(labels)).shape))
    
        #List to keep track of the shapelets in each cluster
        clusters_shapelets = []
        for i in range(n_c):
            clusters_shapelets.append(list([]))
        
        #Group the indices of each cluster (to match previous format)
        for i, label in enumerate(labels):
            clusters_shapelets[label].append(i)
            
        print('Before: ' + str(len(clusters_shapelets))) 
            
        #Remove empty clusters
        i=0
        while i < len(clusters_shapelets):
            if len(clusters_shapelets[i])==0:
                clusters_shapelets.pop(i)
                i = i-1
            i += 1
                
        print('After: ' + str(len(clusters_shapelets)))
            
        #Sort the shapelets indices in each cluster by their score, descendingly
        ##TODO if needed (it is done in previous version)
        
        all_clusters_shapelets.append(np.asarray(clusters_shapelets))
        
    return all_clusters_shapelets

#Z-normalization as a whole, not by column
@jit
def normalize(a):
    if a.std() == 0:
        return np.zeros(len(a))
    else:
        return (a-a.mean()) / a.std()

def cluster_distance_dtw_(X_train, c1, c2, dim):
    distance = 0.0
    num_comparisons = 0

    for s1 in c1:
        idx_1 = s1[0]
        start_1 = s1[1]
        end_1 = s1[2]

        sh_1 = X_train.iloc[idx_1][dim][start_1:end_1]
        sh_1 = normalize(np.asarray(sh_1))

        for s2 in c2:
            idx_2 = s2[0]
            start_2 = s2[1]
            end_2 = s2[2]

            sh_2 = X_train.iloc[idx_2][dim][start_2:end_2]
            sh_2 = normalize(np.asarray(sh_2))

            distance = dtw(s1.flatten(), s2.flatten())
            num_comparisons += 1


def cluster_distance_dtw(shapelets, c1, c2, dim):
    distance = 0.0
    num_comparisons = 0

    for s1 in c1:
        for s2 in c2:
            distance += dtw(shapelets[s1].flatten(), shapelets[s2].flatten())
            num_comparisons += 1

    return distance/num_comparisons

# Cluster similar shapelets of same lenght until the number of clusters at each
# dimension equals n_dim
def cluster_shapelets_hills_dtw(shapelets, n_c):
    # List to hold the shapelets in each cluster, across all dimensions
    all_clusters_shapelets = []

    # Work on each dimension independently
    for dim in range(len(shapelets)):
        # List to keep track of the shapelets in each cluster
        clusters_shapelets = []
        for i in range(len(shapelets[dim])):
            clusters_shapelets.append(list([i]))

        # Until the final number of clusters desired
        while len(clusters_shapelets) > n_c[dim]:
            # Distances matrix
            distances = np.empty((len(clusters_shapelets), len(clusters_shapelets)))

            # Get the distance between every two clusters
            for i in range(len(clusters_shapelets)):
                for j in range(len(clusters_shapelets)):
                    # If same cluster, high distance
                    if i == j:
                        distances[i][j] = np.inf
                    else:
                        distances[i][j] = cluster_distance_dtw(shapelets[dim], clusters_shapelets[i], clusters_shapelets[j], dim)

            # Get the indices of the closest clusters
            ij = np.unravel_index(distances.argmin(), distances.shape)

            # Group the two clusters
            for sh in clusters_shapelets[ij[1]]:
                clusters_shapelets[ij[0]].append(sh)
            del clusters_shapelets[ij[1]]
            
        all_clusters_shapelets.append(np.asarray(clusters_shapelets))

    return all_clusters_shapelets

# Cluster similar shapelets of same lenght until the number of clusters at each
# dimension equals n_dim
def cluster_shapelets_hills_dtw_(X_train, indices, scores, n_dim, n_sh, n_c):
    # List to keep track off the major shapelets of the clusters, all dimensions    
    all_clusters_heads = []

    # List to hold the shapelets in each cluster, across all dimensions
    all_clusters_shapelets = []

    # Work on each dimension independently
    for dim in range(n_dim):
        # Put the shapelet indices in individual lists (clusters_heads)
        clusters_heads = []
        for index in indices[dim]:
            clusters_heads.append(list([index]))

        # List to keep track of the shapelets in each cluster
        clusters_shapelets = []
        for i in range(n_sh[dim]):
            clusters_shapelets.append(list([i]))

        # Until the final number of clusters desired
        while len(clusters_heads) > n_c[dim]:
            # Distances matrix
            distances = np.empty((len(clusters_heads), len(clusters_heads)))

            # Get the distance between every two clusters
            for i in range(len(clusters_heads)):
                for j in range(len(clusters_heads)):
                    # If same cluster, high distance
                    if i == j:
                        distances[i][j] = np.inf
                    else:
                        distances[i][j] = cluster_distance_dtw(
                            X_train, clusters_heads[i], clusters_heads[j], dim)

            # Get the indices of the closest clusters
            ij = np.unravel_index(distances.argmin(), distances.shape)

            # Group the two clusters and remove the one with the lowest score
            if scores[dim][ij[0]] > scores[dim][ij[1]]:
                for sh in clusters_heads[ij[1]]:
                    clusters_heads[ij[0]].append(sh)
                for sh in clusters_shapelets[ij[1]]:
                    clusters_shapelets[ij[0]].append(sh)
                del clusters_heads[ij[1]]
                del scores[dim][ij[1]]
                del clusters_shapelets[ij[1]]

            else:
                for sh in clusters_heads[ij[0]]:  
                    clusters_heads[ij[1]].append(sh)
                for sh in clusters_shapelets[ij[0]]:
                    clusters_shapelets[ij[1]].append(sh)
                del clusters_heads[ij[0]]
                del scores[dim][ij[0]]
                del clusters_shapelets[ij[0]]

            # print(clusters_shapelets)

        all_clusters_heads.append(np.asarray(clusters_heads))
        all_clusters_shapelets.append(np.asarray(clusters_shapelets))

    return all_clusters_shapelets, all_clusters_heads

#Cluster shapelets based on a clustering ration and merge their shapelets_distances according 
#to single shapelet's format
def cluster_shapelets_and_format_kmeans(shapelets, shapelets_distances, shapelets_distances_test, cluster_ratio):
    #The number of clusters per dimension
    n_clusters = []

    all_n_shapelets = []
    
    for shapelet in shapelets:
        n_shapelets = len(shapelet)
        n_clusters.append(int(n_shapelets*cluster_ratio/100))
        all_n_shapelets.append(n_shapelets)


    if all(i < 10 for i in all_n_shapelets):
        print("Less than 10 shapelets per dimension. Not clustering.")
        return
    
    #Cluster
    all_clusters_shapelets = cluster_shapelets_kmeans(shapelets, n_clusters)
    
    #List to hold the distances between each cluster-shapelet and its potential 
    #occurences, for all clusters at all dimensions
    clusters_distances = []
    clusters_distances_test = []
    
    #Merge shapelets_distances of clusters
    for dim in range(len(all_clusters_shapelets)):  
        #List to hold the distances between each cluster-shapelet and its potential 
        #occurences, for all clusters at this dimension
        dim_cluster_distances = []
        dim_cluster_distances_test = []
        
        for c in range(len(all_clusters_shapelets[dim])):
            #List to hold the distances between this cluster-shapelet and its 
            #potential occurences
            cluster_distances = []
            cluster_distances_test = []
            
            for sh in all_clusters_shapelets[dim][c]:
                ##j is the dataset sample index
                j=0
                #Since not all dimensions have the same number of shapelets:
                if sh >= len(shapelets_distances[dim]):
                    print("not all dimensions have the same number of shapelets")
                    #continue
                #print(sh, len(shapelets_distances[dim]), dim)
                for sd in shapelets_distances[dim][sh]:
                    cluster_distances.append([j,np.asarray(sd)])
                    j=j+1
                j=0
                for sd in shapelets_distances_test[dim][sh]:
                    cluster_distances_test.append([j,np.asarray(sd)])
                    j=j+1
                    
            dim_cluster_distances.append(np.asarray(cluster_distances))
            dim_cluster_distances_test.append(np.asarray(cluster_distances_test))
            
        clusters_distances.append(np.asarray(dim_cluster_distances))
        clusters_distances_test.append(np.asarray(dim_cluster_distances_test))
        
    return all_clusters_shapelets, clusters_distances, clusters_distances_test
    
#Cluster shapelets based on a clustering ration and merge their shapelets_distances according 
#to single shapelet's format
def cluster_shapelets_and_format_hills_dtw(shapelets, shapelets_distances, shapelets_distances_test, cluster_ratio):
    #List containting the desired number of clusters per dimension
    n_clusters = []
    
    for sd in shapelets_distances:
        n_shapelets = len(sd)
        n_clusters.append(int(n_shapelets*cluster_ratio/100))

    # if all(i < 10 for i in all_n_shapelets):
    #     print("Less than 10 shapelets per dimension. Not clustering.")
    #     return
    
    #Cluster
    all_clusters_shapelets = cluster_shapelets_hills_dtw(shapelets, n_clusters)
    
    #List to hold the distances between each cluster-shapelet and its potential 
    #occurences, for all clusters at all dimensions
    clusters_distances = []
    clusters_distances_test = []
    
    #Merge shapelets_distances of clusters
    for dim in range(len(all_clusters_shapelets)):  
        #List to hold the distances between each cluster-shapelet and its potential 
        #occurences, for all clusters at this dimension
        dim_cluster_distances = []
        dim_cluster_distances_test = []
        
        for c in range(len(all_clusters_shapelets[dim])):
            #List to hold the distances between this cluster-shapelet and its 
            #potential occurences
            cluster_distances = []
            cluster_distances_test = []
            
            for sh in all_clusters_shapelets[dim][c]:
                ##j is the dataset sample index
                j=0
                #Since not all dimensions have the same number of shapelets:
                # if sh >= len(shapelets_distances[dim]):
                #     print("not all dimensions have the same number of shapelets")
                    #continue
                #print(sh, len(shapelets_distances[dim]), dim)
                for sd in shapelets_distances[dim][sh]:
                    cluster_distances.append([j,np.asarray(sd)])
                    j=j+1
                j=0
                for sd in shapelets_distances_test[dim][sh]:
                    cluster_distances_test.append([j,np.asarray(sd)])
                    j=j+1
                    
            dim_cluster_distances.append(np.asarray(cluster_distances))
            dim_cluster_distances_test.append(np.asarray(cluster_distances_test))
            
        clusters_distances.append(np.asarray(dim_cluster_distances))
        clusters_distances_test.append(np.asarray(dim_cluster_distances_test))
        
    return all_clusters_shapelets, clusters_distances, clusters_distances_test

#Returns the threshold used to select shapelet occurences based on a given percentage
def get_occurences_threshold(shapelets_distances, ts_length, percentage):
    #List to hold all distances values
    sds = []
    
    #Append the scaled distances
    for dim in shapelets_distances:
        for shapelet in dim:
            #Compute the length of the shapelet
            shapelet_length = ts_length - shapelet.shape[1] + 1
            for instance in np.asarray(list(zip(*shapelet))[1]):
                sds.append(instance/shapelet_length)
                
    #Sort the distances ascendingly
    sds.sort()
    
    #Number of shapelet occurences to keep (per shapelet)
    n = int(percentage*len(sds)) 
    
    print(percentage, len(sds), sds[n])
    print('Number of shapelet occurences to keep (per shapelet)',n)
    
    #Return the threshold distance to select the shapelet occurences to keep        
    return sds[n]

#Returns an array with size=n_samples (should be =X_train.shape[0]) where each
#sample value represents the number of times a>b happens
# def get_a_podmsfe_b_count_allen(n_samples, all_shapelet_locations, dim_a, k_a, dim_b, k_b):   
#     #If one of these positions doesn't contain a shapelet, flag it and skip
#     if len(all_shapelet_locations[dim_a]) <= k_a or len(all_shapelet_locations[dim_b]) <= k_b: 
#         no_count = np.full((n_samples), -1, dtype=np.int32)
#         no_index = np.full(n_samples, -1, dtype=np.object)
                       
#         return no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index
     
#     rule_count_p = np.zeros(n_samples, dtype=np.int32)
#     rule_index_p =  np.empty(n_samples, dtype=np.object)    
#     rule_count_o = np.zeros(n_samples, dtype=np.int32)
#     rule_index_o =  np.empty(n_samples, dtype=np.object)
#     rule_count_d = np.zeros(n_samples, dtype=np.int32)
#     rule_index_d =  np.empty(n_samples, dtype=np.object)
#     rule_count_m = np.zeros(n_samples, dtype=np.int32)
#     rule_index_m =  np.empty(n_samples, dtype=np.object)
#     rule_count_s = np.zeros(n_samples, dtype=np.int32)
#     rule_index_s =  np.empty(n_samples, dtype=np.object)
#     rule_count_f = np.zeros(n_samples, dtype=np.int32)
#     rule_index_f =  np.empty(n_samples, dtype=np.object)
#     rule_count_e = np.zeros(n_samples, dtype=np.int32)
#     rule_index_e =  np.empty(n_samples, dtype=np.object)
    
#     for i in range(rule_index_p.shape[0]):
#         rule_index_p[i] = []
#         rule_index_o[i] = []
#         rule_index_d[i] = []
#         rule_index_m[i] = []
#         rule_index_s[i] = []
#         rule_index_f[i] = []
#         rule_index_e[i] = []
    
#     a = all_shapelet_locations[dim_a][k_a]
#     b = all_shapelet_locations[dim_b][k_b]
            
#     a_loc_num = 0
#     a_prev_idx = -99
    
#     for idx, start, end in a:
#         a_prev_idx = idx
#         if a_prev_idx == idx:
#             a_loc_num += 1
#         else:
#             a_loc_num = 0
        
#         #If shapelet doesn't happen (uint32 = -1)
#         if idx == 4294967295:
#             continue
#         hits = b[np.where(b[:,0]==idx)]
        
#         b_loc_num = 0
            
#         for h_idx, h_start, h_end in hits:
#             b_loc_num += 1
            
#             #If shapelet doesn't happen
#             if h_idx == 4294967295:
#                 continue
#             if (h_start>end):
#                 rule_count_p[idx]+=1
#                 rule_index_p[idx].append((start, end, h_start, h_end))
#             elif (start<h_start and end>h_start and end<h_end):
#                 rule_count_o[idx]+=1
#                 rule_index_o[idx].append((start, end, h_start, h_end))
#             elif (start>h_start and end<h_end):
#                 rule_count_d[idx]+=1
#                 rule_index_d[idx].append((start, end, h_start, h_end))
#             elif (start<h_start and end==h_start and end<h_end):
#                 rule_count_m[idx]+=1
#                 rule_index_m[idx].append((start, end, h_start, h_end))
#             elif (start==h_start and end<h_end):
#                 rule_count_s[idx]+=1
#                 rule_index_s[idx].append((start, end, h_start, h_end))  
#             elif (start>h_start and end==h_end):
#                 rule_count_f[idx]+=1
#                 rule_index_f[idx].append((start, end, h_start, h_end))
#             elif (start==h_start and end==h_end and (dim_a!=dim_b or k_a!=k_b)):
#                 rule_count_e[idx]+=1
#                 rule_index_e[idx].append((start, end, h_start, h_end))
#     return rule_count_p, rule_index_p, rule_count_o, rule_index_o, rule_count_d,\
#         rule_index_d, rule_count_m, rule_index_m, rule_count_s, rule_index_s,\
#         rule_count_f, rule_index_f, rule_count_e, rule_index_e


#Returns an array with size=n_samples (should be =X_train.shape[0]) where each
#sample value represents the number of times a>b happens
def get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, k_a, dim_b, k_b):   
    #If one of these positions doesn't contain a shapelet, flag it and skip
    if len(all_shapelet_locations[dim_a]) <= k_a or len(all_shapelet_locations[dim_b]) <= k_b: 
        no_count = np.full((n_samples), -1, dtype=np.int32)
        no_index = np.full(n_samples, -1, dtype=np.object)
                       
        return no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index
    
    
    rule_count_p = np.zeros(n_samples, dtype=np.int32)
    rule_index_p =  np.empty(n_samples, dtype=np.object)    
    rule_count_o = np.zeros(n_samples, dtype=np.int32)
    rule_index_o =  np.empty(n_samples, dtype=np.object)
    rule_count_d = np.zeros(n_samples, dtype=np.int32)
    rule_index_d =  np.empty(n_samples, dtype=np.object)
    rule_count_m = np.zeros(n_samples, dtype=np.int32)
    rule_index_m =  np.empty(n_samples, dtype=np.object)
    rule_count_s = np.zeros(n_samples, dtype=np.int32)
    rule_index_s =  np.empty(n_samples, dtype=np.object)
    rule_count_f = np.zeros(n_samples, dtype=np.int32)
    rule_index_f =  np.empty(n_samples, dtype=np.object)
    rule_count_e = np.zeros(n_samples, dtype=np.int32)
    rule_index_e =  np.empty(n_samples, dtype=np.object)
    
    for i in range(rule_index_p.shape[0]):
        rule_index_p[i] = []
        rule_index_o[i] = []
        rule_index_d[i] = []
        rule_index_m[i] = []
        rule_index_s[i] = []
        rule_index_f[i] = []
        rule_index_e[i] = []
    
    a = all_shapelet_locations[dim_a][k_a]
    b = all_shapelet_locations[dim_b][k_b]
    
    # print("DIMAKA ", dim_a,k_a, len(all_shapelet_locations), len(all_shapelet_locations[0]))
        
    for idx, start, end in a:
        # print(ise)
        # idx, start, end = ise
        # print(dim_a,k_a,dim_b,k_b)
        # print(idx,start,end)
        #If shapelet doesn't happen (uint32 = -1)
        if idx == 4294967295:
            continue
        hits = b[np.where(b[:,0]==idx)]
            
        for h_idx, h_start, h_end in hits:
            # print('hits: ', h_idx,h_start,h_end)
            #If shapelet doesn't happen
            if h_idx == 4294967295:
                continue
            if (h_start>end):
                rule_count_p[idx]+=1
                rule_index_p[idx].append((start, end, h_start, h_end))
            elif (start<h_start and end>h_start and end<h_end):
                rule_count_o[idx]+=1
                rule_index_o[idx].append((start, end, h_start, h_end))
            elif (start>h_start and end<h_end):
                rule_count_d[idx]+=1
                rule_index_d[idx].append((start, end, h_start, h_end))
            elif (start<h_start and end==h_start and end<h_end):
                rule_count_m[idx]+=1
                rule_index_m[idx].append((start, end, h_start, h_end))
            elif (start==h_start and end<h_end):
                rule_count_s[idx]+=1
                rule_index_s[idx].append((start, end, h_start, h_end))  
            elif (start>h_start and end==h_end):
                rule_count_f[idx]+=1
                rule_index_f[idx].append((start, end, h_start, h_end))
            elif (start==h_start and end==h_end and (dim_a!=dim_b or k_a!=k_b)):
                rule_count_e[idx]+=1
                rule_index_e[idx].append((start, end, h_start, h_end))
    return rule_count_p, rule_index_p, rule_count_o, rule_index_o, rule_count_d,\
        rule_index_d, rule_count_m, rule_index_m, rule_count_s, rule_index_s,\
        rule_count_f, rule_index_f, rule_count_e, rule_index_e


#Returns 7 3D arrays where for each shapelet b at [dim][k], the number of times
#a->b happens in each sample is provided (+ 6 arrays for the indices). n_samples should be =X_train.shape[0]
@njit
def get_all_a_podmsfe_bs_count(n_samples, all_shapelet_locations, n_dims, n_k, dim_a, k_a):  
    
    rules_count_p = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_p = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_count_o = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_o = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_count_d = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_d = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_count_m = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_m = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_count_s = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_s = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_count_f = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_f = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_count_e = np.zeros((n_dims, n_k, n_samples), dtype=np.int32)
    rules_index_e = np.zeros((n_dims, n_k, n_samples, 4), dtype=np.uint16)

    
    for dim in range(n_dims):
        for k in range(n_k):
            # print("a_bs")
            # print(dim)
            # print(k)
            # print("---")
    
            rules_count_p[dim][k], rules_index_p[dim][k], rules_count_o[dim][k], rules_index_o[dim][k], rules_count_d[dim][k], rules_index_d[dim][k], rules_count_m[dim][k], rules_index_m[dim][k], rules_count_s[dim][k], rules_index_s[dim][k], rules_count_f[dim][k], rules_index_f[dim][k], rules_count_e[dim][k], rules_index_e[dim][k] = get_a_podmsfe_b_count(n_samples, all_shapelet_locations, dim_a, k_a, dim, k)

    return rules_count_p, rules_index_p, rules_count_o, rules_index_o, rules_count_d, rules_index_d, rules_count_m, rules_index_m, rules_count_s, rules_index_s, rules_count_f, rules_index_f, rules_count_e, rules_index_e


#Returns 7 5D arrays (one for each rule type) where all a>b possible rules are counted. The first and 
#second dimensions represent the [dim][k] of the antecedent, the third and fourth
#dimensions represent the [dim][k] of the subsequent, and the fifth dimension
#contains an array where the number of times a->b happens in each sample is 
@njit
def get_all_as_podmsfe_bs_count(n_samples, all_shapelet_locations):
    #Get the number of shapelets in the dimension with the most shapelets
    n_k = 0
    for i in range(len(all_shapelet_locations)):
        this_length = len(all_shapelet_locations[i])
        if  this_length > n_k:
            n_k = this_length
    
    n_dims = len(all_shapelet_locations)
    
    rules_counts_p = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_p = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_counts_o = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_o = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_counts_d = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_d = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_counts_m = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_m = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_counts_s = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_s = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_counts_f = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_f = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
    rules_counts_e = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_e = np.zeros((n_dims, n_k, n_dims, n_k, n_samples, 4), dtype=np.uint16)
        
    print("Hloo")
    
    for dim in range(n_dims):
        for k in range(n_k):
            # print("as_bs")
            # print(dim)
            # print(k)
            # print("---")
            rules_counts_p[dim][k], rules_indices_p[dim][k], rules_counts_o[dim][k], rules_indices_o[dim][k], rules_counts_d[dim][k], rules_indices_d[dim][k], rules_counts_m[dim][k], rules_indices_m[dim][k], rules_counts_s[dim][k], rules_indices_s[dim][k], rules_counts_f[dim][k], rules_indices_f[dim][k], rules_counts_e[dim][k], rules_indices_e[dim][k] = get_all_a_podmsfe_bs_count(n_samples, all_shapelet_locations, n_dims, n_k, dim, k)
    
    return rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d, rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e
   
def get_all_as_podmsfe_bs_count_contracted_1(n_samples, all_shapelet_locations, time_contract):
    #for timing the counting when contracting
    start_time = time.time()        
    
    #Get the number of shapelets in the dimension with the most shapelets
    n_k = 0
    for i in range(len(all_shapelet_locations)):
        this_length = len(all_shapelet_locations[i])
        if  this_length > n_k:
            n_k = this_length
    
    n_dims = len(all_shapelet_locations)
    
    rules_counts_p = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_p = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_o = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_o = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_d = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_d = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_m = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_m = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_s = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_s = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_f = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_f = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_e = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_e = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    
    rules_counts_p_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_o_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_d_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_m_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_s_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_f_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_e_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
        
    
    print("Hloo")
    
    #Looping through the shapelet pairs in a "BFS" like manner. Making sure that 
    #rules between stronger shapelets are mined first
    
    #Flag to keep track of time contract
    time_ended = False
    
    #List to save indices of mined rules
    mined_rules = [[],[],[],[]]
    
    for k in range(n_k):
        if time_ended:
            break
        for dim_a in range(n_dims):
            if time_ended:
                break
            for dim_b in range(n_dims): 
                if time_ended:
                    break
                for s in range(k+1):
                    if time_ended:
                        break
                    #If time contract ended, return
                    if time.time()-start_time>=time_contract*60:
                        time_ended = True
                                                
                    #Count a>b
                    rules_counts_p[dim_a][k][dim_b][s], rules_indices_p[dim_a][k][dim_b][s], rules_counts_o[dim_a][k][dim_b][s], rules_indices_o[dim_a][k][dim_b][s], rules_counts_d[dim_a][k][dim_b][s], rules_indices_d[dim_a][k][dim_b][s], rules_counts_m[dim_a][k][dim_b][s], rules_indices_m[dim_a][k][dim_b][s], rules_counts_s[dim_a][k][dim_b][s], rules_indices_s[dim_a][k][dim_b][s], rules_counts_f[dim_a][k][dim_b][s], rules_indices_f[dim_a][k][dim_b][s], rules_counts_e[dim_a][k][dim_b][s], rules_indices_e[dim_a][k][dim_b][s] = get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, k, dim_b, s)
                    rules_counts_p_[dim_a][k][dim_b][s], rules_counts_o_[dim_a][k][dim_b][s], rules_counts_d_[dim_a][k][dim_b][s], rules_counts_m_[dim_a][k][dim_b][s], rules_counts_s_[dim_a][k][dim_b][s], rules_counts_f_[dim_a][k][dim_b][s], rules_counts_e_[dim_a][k][dim_b][s] = np.array([dim_a, k, dim_b, s], dtype=np.int32), np.array([dim_a, k, dim_b, s], dtype=np.int32), np.array([dim_a, k, dim_b, s], dtype=np.int32), np.array([dim_a, k, dim_b, s], dtype=np.int32), np.array([dim_a, k, dim_b, s], dtype=np.int32), np.array([dim_a, k, dim_b, s], dtype=np.int32), np.array([dim_a, k, dim_b, s], dtype=np.int32)
                    
                    #Save the mined indices
                    mined_rules[0].append(dim_a)
                    mined_rules[1].append(k)
                    mined_rules[2].append(dim_b)
                    mined_rules[3].append(s)
                    
                    #Count b>a unless a==b
                    if s!=k:
                        rules_counts_p[dim_a][s][dim_b][k], rules_indices_p[dim_a][s][dim_b][k], rules_counts_o[dim_a][s][dim_b][k], rules_indices_o[dim_a][s][dim_b][k], rules_counts_d[dim_a][s][dim_b][k], rules_indices_d[dim_a][s][dim_b][k], rules_counts_m[dim_a][s][dim_b][k], rules_indices_m[dim_a][s][dim_b][k], rules_counts_s[dim_a][s][dim_b][k], rules_indices_s[dim_a][s][dim_b][k], rules_counts_f[dim_a][s][dim_b][k], rules_indices_f[dim_a][s][dim_b][k], rules_counts_e[dim_a][s][dim_b][k], rules_indices_e[dim_a][s][dim_b][k] = get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, s, dim_b, k)
                        rules_counts_p_[dim_a][s][dim_b][k], rules_counts_o_[dim_a][s][dim_b][k], rules_counts_d_[dim_a][s][dim_b][k], rules_counts_m_[dim_a][s][dim_b][k], rules_counts_s_[dim_a][s][dim_b][k], rules_counts_f_[dim_a][s][dim_b][k], rules_counts_e_[dim_a][s][dim_b][k], = np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32)

                        mined_rules[0].append(dim_a)
                        mined_rules[1].append(s)
                        mined_rules[2].append(dim_b)
                        mined_rules[3].append(k)
                    
    #Keep only mined rules
    # rules_counts_p = rules_counts_p[mined_rules]
    # rules_indices_p = rules_indices_p[mined_rules]
    # rules_counts_o = rules_counts_o[mined_rules]
    # rules_indices_o = rules_indices_o[mined_rules]
    # rules_counts_d = rules_counts_d[mined_rules]
    # rules_indices_d = rules_indices_d[mined_rules]
    # rules_counts_m = rules_counts_m[mined_rules]
    # rules_indices_m = rules_indices_m[mined_rules]
    # rules_counts_s = rules_counts_s[mined_rules]
    # rules_indices_s = rules_indices_s[mined_rules]
    # rules_counts_f = rules_counts_f[mined_rules]
    # rules_indices_f = rules_indices_f[mined_rules]
    # rules_counts_e = rules_counts_e[mined_rules]
    # rules_indices_e = rules_indices_e[mined_rules]
    
    if time_ended:
        print("Time is up! Couldn't go through all shapelet combinations.")
    
    return mined_rules, rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d,\
        rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e,\
        rules_counts_p_, rules_counts_o_, rules_counts_d_, rules_counts_m_, rules_counts_s_, rules_counts_f_, rules_counts_e_

def get_all_as_podmsfe_bs_count_contracted_11(n_samples, all_shapelet_locations, time_contract):
    #for timing the counting when contracting
    start_time = time.time()        
    
    #Get the number of shapelets in the dimension with the most shapelets
    n_k = 0
    for i in range(len(all_shapelet_locations)):
        this_length = len(all_shapelet_locations[i])
        if  this_length > n_k:
            n_k = this_length
    
    n_dims = len(all_shapelet_locations)
    
    rules_counts_p = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_p = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_o = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_o = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_d = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_d = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_m = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_m = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_s = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_s = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_f = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_f = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_e = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_e = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    
    rules_counts_p_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_o_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_d_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_m_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_s_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_f_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_counts_e_ = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
        
    
    print("Hloo")
    
    #Looping through the shapelet pairs in a "BFS" like manner. Making sure that 
    #rules between stronger shapelets are mined first
    
    #Flag to keep track of time contract
    time_ended = False
    
    #List to save indices of mined rules
    mined_rules = [[],[],[],[]]
    
    for k in range(n_k):
        if time_ended:
            break
        for dim_a in range(n_dims):
            if time_ended:
                break
            for dim_b in range(n_dims): 
                if time_ended:
                    break
                for s in range(k+1):
                    if time_ended:
                        break
                    #If time contract ended, return
                    if time.time()-start_time>=time_contract*60:
                        time_ended = True
                                                
                    #Count a>b
                    rules_counts_p[dim_a][k][dim_b][s], rules_indices_p[dim_a][k][dim_b][s],\
                        rules_counts_o[dim_a][k][dim_b][s], rules_indices_o[dim_a][k][dim_b][s],\
                        rules_counts_d[dim_a][k][dim_b][s], rules_indices_d[dim_a][k][dim_b][s],\
                        rules_counts_m[dim_a][k][dim_b][s], rules_indices_m[dim_a][k][dim_b][s],\
                        rules_counts_s[dim_a][k][dim_b][s], rules_indices_s[dim_a][k][dim_b][s],\
                        rules_counts_f[dim_a][k][dim_b][s], rules_indices_f[dim_a][k][dim_b][s],\
                        rules_counts_e[dim_a][k][dim_b][s], rules_indices_e[dim_a][k][dim_b][s] =\
                        get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, k, dim_b, s)
                    rules_counts_p_[dim_a][k][dim_b][s], rules_counts_o_[dim_a][k][dim_b][s],\
                        rules_counts_d_[dim_a][k][dim_b][s], rules_counts_m_[dim_a][k][dim_b][s],\
                        rules_counts_s_[dim_a][k][dim_b][s], rules_counts_f_[dim_a][k][dim_b][s],\
                        rules_counts_e_[dim_a][k][dim_b][s] = np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                        np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                        np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                        np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                        np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                        np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                        np.array([dim_a, k, dim_b, s], dtype=np.int32)
                    
                    #Save the mined indices
                    if not (rules_counts_p[dim_a][k][dim_b][s][0]==1 or rules_counts_p[dim_a][k][dim_b][s][0]==4294967295 or np.sum(rules_counts_p[dim_a][k][dim_b][s])==0):
                        mined_rules[0].append(dim_a)
                        mined_rules[1].append(k)
                        mined_rules[2].append(dim_b)
                        mined_rules[3].append(s)
                    
                    #Count b>a unless a==b
                    if s!=k:
                        rules_counts_p[dim_a][s][dim_b][k], rules_indices_p[dim_a][s][dim_b][k], rules_counts_o[dim_a][s][dim_b][k], rules_indices_o[dim_a][s][dim_b][k], rules_counts_d[dim_a][s][dim_b][k], rules_indices_d[dim_a][s][dim_b][k], rules_counts_m[dim_a][s][dim_b][k], rules_indices_m[dim_a][s][dim_b][k], rules_counts_s[dim_a][s][dim_b][k], rules_indices_s[dim_a][s][dim_b][k], rules_counts_f[dim_a][s][dim_b][k], rules_indices_f[dim_a][s][dim_b][k], rules_counts_e[dim_a][s][dim_b][k], rules_indices_e[dim_a][s][dim_b][k] = get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, s, dim_b, k)
                        rules_counts_p_[dim_a][s][dim_b][k], rules_counts_o_[dim_a][s][dim_b][k], rules_counts_d_[dim_a][s][dim_b][k], rules_counts_m_[dim_a][s][dim_b][k], rules_counts_s_[dim_a][s][dim_b][k], rules_counts_f_[dim_a][s][dim_b][k], rules_counts_e_[dim_a][s][dim_b][k], = np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32)
                        
                        if not (rules_counts_p[dim_a][s][dim_b][k][0]==1 or rules_counts_p[dim_a][s][dim_b][k][0]==4294967295 or np.sum(rules_counts_p[dim_a][s][dim_b][k])==0):
                            mined_rules[0].append(dim_a)
                            mined_rules[1].append(s)
                            mined_rules[2].append(dim_b)
                            mined_rules[3].append(k)
                    
    #Keep only mined rules
    # rules_counts_p = rules_counts_p[mined_rules]
    # rules_indices_p = rules_indices_p[mined_rules]
    # rules_counts_o = rules_counts_o[mined_rules]
    # rules_indices_o = rules_indices_o[mined_rules]
    # rules_counts_d = rules_counts_d[mined_rules]
    # rules_indices_d = rules_indices_d[mined_rules]
    # rules_counts_m = rules_counts_m[mined_rules]
    # rules_indices_m = rules_indices_m[mined_rules]
    # rules_counts_s = rules_counts_s[mined_rules]
    # rules_indices_s = rules_indices_s[mined_rules]
    # rules_counts_f = rules_counts_f[mined_rules]
    # rules_indices_f = rules_indices_f[mined_rules]
    # rules_counts_e = rules_counts_e[mined_rules]
    # rules_indices_e = rules_indices_e[mined_rules]
    
    if time_ended:
        print("Time is up! Couldn't go through all shapelet combinations.")
    
    print('rrrr', rules_counts_p.shape, rules_counts_p[mined_rules].shape)

    return mined_rules, rules_counts_p[mined_rules], rules_indices_p[mined_rules], rules_counts_o[mined_rules], rules_indices_o[mined_rules], rules_counts_d[mined_rules], rules_indices_d[mined_rules],\
        rules_counts_m[mined_rules], rules_indices_m[mined_rules], rules_counts_s[mined_rules], rules_indices_s[mined_rules], rules_counts_f[mined_rules], rules_indices_f[mined_rules], rules_counts_e[mined_rules], rules_indices_e[mined_rules],\
        rules_counts_p_[mined_rules], rules_counts_o_[mined_rules], rules_counts_d_[mined_rules], rules_counts_m_[mined_rules], rules_counts_s_[mined_rules], rules_counts_f_[mined_rules], rules_counts_e_[mined_rules]
    
   
#Returns 6 5D arrays (one for each rule type) where all a>b possible rules are counted. The first and 
#second dimensions represent the [dim][k] of the antecedent, the third and fourth
#dimensions represent the [dim][k] of the subsequent, and the fifth dimension
#contains an array where the number of times a->b happens in each sample is 
#provided. n_samples should be =X_train.shape[0]
#Contracted version
def get_all_as_podmsfe_bs_count_contracted(n_samples, all_shapelet_locations, time_contract):
    #for timing the counting when contracting
    start_time = time.time()        
    
    #Get the number of shapelets in the dimension with the most shapelets
    n_k = 0
    for i in range(len(all_shapelet_locations)):
        this_length = len(all_shapelet_locations[i])
        if  this_length > n_k:
            n_k = this_length
    
    n_dims = len(all_shapelet_locations)
    
    rules_counts_p = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_p = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_o = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_o = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_d = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_d = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_m = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_m = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_s = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_s = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_f = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_f = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    rules_counts_e = np.zeros((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.int32)
    rules_indices_e = np.empty((n_dims, n_k, n_dims, n_k, n_samples), dtype=np.object)
    
    print("Hloo")
    
    #Looping through the shapelet pairs in a "BFS" like manner. Making sure that 
    #rules between stronger shapelets are mined first
    
    #Flag to keep track of time contract
    time_ended = False
    
    #List to save indices of mined rules
    mined_rules = [[],[],[],[]]
    
    for k in range(n_k):
        if time_ended:
            break
        for dim_a in range(n_dims):
            if time_ended:
                break
            for dim_b in range(n_dims): 
                if time_ended:
                    break
                for s in range(k+1):
                    if time_ended:
                        break
                    #If time contract ended, return
                    if time.time()-start_time>=time_contract*60:
                        time_ended = True
                                                
                    #Count a>b
                    rules_counts_p[dim_a][k][dim_b][s], rules_indices_p[dim_a][k][dim_b][s], rules_counts_o[dim_a][k][dim_b][s], rules_indices_o[dim_a][k][dim_b][s], rules_counts_d[dim_a][k][dim_b][s], rules_indices_d[dim_a][k][dim_b][s], rules_counts_m[dim_a][k][dim_b][s], rules_indices_m[dim_a][k][dim_b][s], rules_counts_s[dim_a][k][dim_b][s], rules_indices_s[dim_a][k][dim_b][s], rules_counts_f[dim_a][k][dim_b][s], rules_indices_f[dim_a][k][dim_b][s], rules_counts_e[dim_a][k][dim_b][s], rules_indices_e[dim_a][k][dim_b][s] = get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, k, dim_b, s)
                    
                    #Save the mined indices
                    mined_rules[0].append(dim_a)
                    mined_rules[1].append(k)
                    mined_rules[2].append(dim_b)
                    mined_rules[3].append(s)
                    
                    #Count b>a unless a==b
                    if s!=k:
                        rules_counts_p[dim_a][s][dim_b][k], rules_indices_p[dim_a][s][dim_b][k], rules_counts_o[dim_a][s][dim_b][k], rules_indices_o[dim_a][s][dim_b][k], rules_counts_d[dim_a][s][dim_b][k], rules_indices_d[dim_a][s][dim_b][k], rules_counts_m[dim_a][s][dim_b][k], rules_indices_m[dim_a][s][dim_b][k], rules_counts_s[dim_a][s][dim_b][k], rules_indices_s[dim_a][s][dim_b][k], rules_counts_f[dim_a][s][dim_b][k], rules_indices_f[dim_a][s][dim_b][k], rules_counts_e[dim_a][s][dim_b][k], rules_indices_e[dim_a][s][dim_b][k] = get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, dim_a, s, dim_b, k)
                    
                        mined_rules[0].append(dim_a)
                        mined_rules[1].append(s)
                        mined_rules[2].append(dim_b)
                        mined_rules[3].append(k)
                        
                    
    #Keep only mined rules
    # rules_counts_p = rules_counts_p[mined_rules]
    # rules_indices_p = rules_indices_p[mined_rules]
    # rules_counts_o = rules_counts_o[mined_rules]
    # rules_indices_o = rules_indices_o[mined_rules]
    # rules_counts_d = rules_counts_d[mined_rules]
    # rules_indices_d = rules_indices_d[mined_rules]
    # rules_counts_m = rules_counts_m[mined_rules]
    # rules_indices_m = rules_indices_m[mined_rules]
    # rules_counts_s = rules_counts_s[mined_rules]
    # rules_indices_s = rules_indices_s[mined_rules]
    # rules_counts_f = rules_counts_f[mined_rules]
    # rules_indices_f = rules_indices_f[mined_rules]
    # rules_counts_e = rules_counts_e[mined_rules]
    # rules_indices_e = rules_indices_e[mined_rules]
    
    if time_ended:
        print("Time is up! Couldn't go through all shapelet combinations.")
    
    return mined_rules, rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d, rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e

#Returns 6 5D arrays (one for each rule type) where all a>b possible rules are counted. The first and 
#second dimensions represent the [dim][k] of the antecedent, the third and fourth
#dimensions represent the [dim][k] of the subsequent, and the fifth dimension
#contains an array where the number of times a->b happens in each sample is 
#provided. n_samples should be =X_train.shape[0]
#For the test set (only process count supports of rules provided)
def get_all_as_podmsfe_bs_count_test(n_samples, all_shapelet_locations, rules_list):    
    #The number of rules
    n_rules = len(rules_list[0]) 
    
    rules_counts_p = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_p = np.zeros((n_rules, n_samples), dtype=np.object)
    rules_counts_o = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_o = np.zeros((n_rules, n_samples), dtype=np.object)
    rules_counts_d = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_d = np.zeros((n_rules, n_samples), dtype=np.object)
    rules_counts_m = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_m = np.zeros((n_rules, n_samples), dtype=np.object)
    rules_counts_s = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_s = np.zeros((n_rules, n_samples), dtype=np.object)
    rules_counts_f = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_f = np.zeros((n_rules, n_samples), dtype=np.object)
    rules_counts_e = np.zeros((n_rules, n_samples), dtype=np.int32)
    rules_indices_e = np.zeros((n_rules, n_samples), dtype=np.object)
        
    print("HlooTest, # rules: " +  str(n_rules))
    
    for r in range(n_rules):
        #print("Rule: " + str(r))
        rules_counts_p[r], rules_indices_p[r], rules_counts_o[r], rules_indices_o[r], rules_counts_d[r], rules_indices_d[r], rules_counts_m[r], rules_indices_m[r], rules_counts_s[r], rules_indices_s[r], rules_counts_f[r], rules_indices_f[r], rules_counts_e[r], rules_indices_e[r] = get_a_podmsfe_b_count_(n_samples, all_shapelet_locations, rules_list[0][r], rules_list[1][r], rules_list[2][r], rules_list[3][r])
        
    return rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d, rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e

