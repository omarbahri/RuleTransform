#Author: Omar Bahri

import numpy as np
from numba import njit, typed
import random
import time

from tslearn.metrics import dtw

from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y
from sklearn.utils import check_random_state
from sktime.transformations.base import _PanelToTabularTransformer

from ruletransform.shapelets import ShapeletTransform, ContractedShapeletTransform

from ruletransform.utils import MultivariateTransformer, get_shapelets, get_shapelets_distances

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

#Returns an array with size=n_samples (should be =X_train.shape[0]) where each
#sample value represents the number of times a>b happens
@njit()
def _get_a_podmsfe_b_count(n_samples, all_shapelet_locations, dim_a, k_a, dim_b, k_b):   
    #If one of these positions doesn't contain a shapelet, flag it and skip
    if len(all_shapelet_locations[dim_a]) <= k_a or len(all_shapelet_locations[dim_b]) <= k_b: 
        no_count = np.full((n_samples), -1, dtype=np.int32)
        no_index = np.full((n_samples, 4), -1, dtype=np.uint16)
                       
        return no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index, no_count, no_index
    
    
    rule_count_p = np.zeros(n_samples, dtype=np.int32)
    rule_index_p =  np.zeros((n_samples, 4), dtype=np.uint16)    
    rule_count_o = np.zeros(n_samples, dtype=np.int32)
    rule_index_o =  np.zeros((n_samples, 4), dtype=np.uint16)
    rule_count_d = np.zeros(n_samples, dtype=np.int32)
    rule_index_d =  np.zeros((n_samples, 4), dtype=np.uint16)
    rule_count_m = np.zeros(n_samples, dtype=np.int32)
    rule_index_m =  np.zeros((n_samples, 4), dtype=np.uint16)
    rule_count_s = np.zeros(n_samples, dtype=np.int32)
    rule_index_s =  np.zeros((n_samples, 4), dtype=np.uint16)
    rule_count_f = np.zeros(n_samples, dtype=np.int32)
    rule_index_f =  np.zeros((n_samples, 4), dtype=np.uint16)
    rule_count_e = np.zeros(n_samples, dtype=np.int32)
    rule_index_e =  np.zeros((n_samples, 4), dtype=np.uint16)
    
    a = all_shapelet_locations[dim_a][k_a]
    b = all_shapelet_locations[dim_b][k_b]
        
    
    for idx, start, end in a:
        #If shapelet doesn't happen (uint32 = -1)
        if idx == 4294967295:
            continue
        hits = b[np.where(b[:,0]==idx)]
            
        for h_idx, h_start, h_end in hits:
            #If shapelet doesn't happen
            if h_idx == 4294967295:
                continue
            if (h_start>end):
                rule_count_p[idx]+=1
                rule_index_p[idx]=[start, end, h_start, h_end]
            elif (start<h_start and end>h_start and end<h_end):
                rule_count_o[idx]+=1
                rule_index_o[idx]=[start, end, h_start, h_end]
            elif (start>h_start and end<h_end):
                rule_count_d[idx]+=1
                rule_index_d[idx]=[start, end, h_start, h_end]
            elif (start<h_start and end==h_start and end<h_end):
                rule_count_m[idx]+=1
                rule_index_m[idx]=[start, end, h_start, h_end]
            elif (start==h_start and end<h_end):
                rule_count_s[idx]+=1
                rule_index_s[idx]=[start, end, h_start, h_end]
            elif (start>h_start and end==h_end):
                rule_count_f[idx]+=1
                rule_index_f[idx]=[start, end, h_start, h_end]
            elif (start==h_start and end==h_end and (dim_a!=dim_b or k_a!=k_b)):
                rule_count_e[idx]+=1
                rule_index_e[idx]=[start, end, h_start, h_end]
    return rule_count_p, rule_index_p, rule_count_o, rule_index_o, rule_count_d, rule_index_d, rule_count_m, rule_index_m, rule_count_s, rule_index_s, rule_count_f, rule_index_f, rule_count_e, rule_index_e

def _get_all_as_podmsfe_bs_count_contracted(n_samples, all_shapelet_locations, time_contract=None, test=False, rules_list=None):
        if not test:
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
            
            rules_shs_p = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
            rules_shs_o = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
            rules_shs_d = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
            rules_shs_m = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
            rules_shs_s = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
            rules_shs_f = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
            rules_shs_e = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
                    
            #Looping through the shapelet pairs in a "BFS" like manner. Making sure that 
            #rules between stronger shapelets are mined first
            
            #Flag to keep track of time contract
            time_ended = False
            
            #List to save indices of mined rules
            mined_rules = [[],[],[],[]]
            
            for k in range(n_k):
                for dim_a in range(n_dims):
                    for dim_b in range(n_dims): 
                        for s in range(k+1):
                            if time_ended:
                                break
                            if not test:
                                #If time contract ended, return
                                if time.time()-start_time>=time_contract*60:
                                    time_ended = True
                                
                            #Count a>b
                            (rules_counts_p[dim_a][k][dim_b][s], rules_indices_p[dim_a][k][dim_b][s],
                            rules_counts_o[dim_a][k][dim_b][s], rules_indices_o[dim_a][k][dim_b][s],
                            rules_counts_d[dim_a][k][dim_b][s], rules_indices_d[dim_a][k][dim_b][s],
                            rules_counts_m[dim_a][k][dim_b][s], rules_indices_m[dim_a][k][dim_b][s],
                            rules_counts_s[dim_a][k][dim_b][s], rules_indices_s[dim_a][k][dim_b][s],
                            rules_counts_f[dim_a][k][dim_b][s], rules_indices_f[dim_a][k][dim_b][s],
                            rules_counts_e[dim_a][k][dim_b][s], rules_indices_e[dim_a][k][dim_b][s])\
                            = _get_a_podmsfe_b_count(n_samples, all_shapelet_locations, dim_a, k, dim_b, s)
                            rules_shs_p[dim_a][k][dim_b][s], rules_shs_o[dim_a][k][dim_b][s],\
                                rules_shs_d[dim_a][k][dim_b][s], rules_shs_m[dim_a][k][dim_b][s],\
                                rules_shs_s[dim_a][k][dim_b][s], rules_shs_f[dim_a][k][dim_b][s],\
                                rules_shs_e[dim_a][k][dim_b][s] = np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                                np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                                np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                                np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                                np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                                np.array([dim_a, k, dim_b, s], dtype=np.int32),\
                                np.array([dim_a, k, dim_b, s], dtype=np.int32)
                                
                            #Count b>a unless a==b
                            if s!=k:
                                (rules_counts_p[dim_a][s][dim_b][k], rules_indices_p[dim_a][s][dim_b][k],
                                rules_counts_o[dim_a][s][dim_b][k], rules_indices_o[dim_a][s][dim_b][k],
                                rules_counts_d[dim_a][s][dim_b][k], rules_indices_d[dim_a][s][dim_b][k],
                                rules_counts_m[dim_a][s][dim_b][k], rules_indices_m[dim_a][s][dim_b][k],
                                rules_counts_s[dim_a][s][dim_b][k], rules_indices_s[dim_a][s][dim_b][k],
                                rules_counts_f[dim_a][s][dim_b][k], rules_indices_f[dim_a][s][dim_b][k],
                                rules_counts_e[dim_a][s][dim_b][k], rules_indices_e[dim_a][s][dim_b][k])\
                                = _get_a_podmsfe_b_count(n_samples, all_shapelet_locations, dim_a, s, dim_b, k)
                                rules_shs_p[dim_a][s][dim_b][k], rules_shs_o[dim_a][s][dim_b][k], rules_shs_d[dim_a][s][dim_b][k], rules_shs_m[dim_a][s][dim_b][k], rules_shs_s[dim_a][s][dim_b][k], rules_shs_f[dim_a][s][dim_b][k], rules_shs_e[dim_a][s][dim_b][k] = np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32), np.array([dim_a, s, dim_b, k], dtype=np.int32)
                                
                            #Save the mined indices
                            mined_rules[0].append(dim_a)
                            mined_rules[1].append(k)
                            mined_rules[2].append(dim_b)
                            mined_rules[3].append(s)
                            
                            mined_rules[0].append(dim_a)
                            mined_rules[1].append(s)
                            mined_rules[2].append(dim_b)
                            mined_rules[3].append(k)
                            
            #Keep only mined rules
            rules_counts_p = rules_counts_p[mined_rules]
            rules_indices_p = rules_indices_p[mined_rules]
            rules_shs_p = rules_shs_p[mined_rules]
            rules_counts_o = rules_counts_o[mined_rules]
            rules_indices_o = rules_indices_o[mined_rules]
            rules_shs_o = rules_shs_o[mined_rules]
            rules_counts_d = rules_counts_d[mined_rules]
            rules_indices_d = rules_indices_d[mined_rules]
            rules_shs_d = rules_shs_d[mined_rules]
            rules_counts_m = rules_counts_m[mined_rules]
            rules_indices_m = rules_indices_m[mined_rules]
            rules_shs_m = rules_shs_m[mined_rules]
            rules_counts_s = rules_counts_s[mined_rules]
            rules_indices_s = rules_indices_s[mined_rules]
            rules_shs_s = rules_shs_s[mined_rules]
            rules_counts_f = rules_counts_f[mined_rules]
            rules_indices_f = rules_indices_f[mined_rules]
            rules_shs_f = rules_shs_f[mined_rules]
            rules_counts_e = rules_counts_e[mined_rules]
            rules_indices_e = rules_indices_e[mined_rules]
            rules_shs_e = rules_shs_e[mined_rules]
            
            if time_ended:
                print("Time is up! Couldn't go through all shapelet combinations.")
            
            return (rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d, rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e, rules_shs_p, rules_shs_o, rules_shs_d, rules_shs_m, rules_shs_s, rules_shs_f, rules_shs_e)

        else:
            #The number of rules
            n_rules = len(rules_list[0]) 
            
            rules_counts_p = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_p = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
            rules_counts_o = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_o = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
            rules_counts_d = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_d = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
            rules_counts_m = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_m = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
            rules_counts_s = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_s = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
            rules_counts_f = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_f = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
            rules_counts_e = np.zeros((n_rules, n_samples), dtype=np.int32)
            rules_indices_e = np.zeros((n_rules, n_samples, 4), dtype=np.uint16)
                            
            for r in range(n_rules):
                #print("Rule: " + str(r))
                (rules_counts_p[r], rules_indices_p[r], rules_counts_o[r], rules_indices_o[r],
                rules_counts_d[r], rules_indices_d[r], rules_counts_m[r], rules_indices_m[r],
                rules_counts_s[r], rules_indices_s[r], rules_counts_f[r], rules_indices_f[r],
                rules_counts_e[r], rules_indices_e[r])\
                = _get_a_podmsfe_b_count(n_samples, all_shapelet_locations, rules_list[0][r], rules_list[1][r], rules_list[2][r], rules_list[3][r])
                
            return (rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d, rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e)
#Returns 7 3D arrays where for each shapelet b at [dim][k], the number of times
#a->b happens in each sample is provided (+ 6 arrays for the indices). n_samples should be =X_train.shape[0]
@njit
def _get_all_a_podmsfe_bs_count(n_samples, all_shapelet_locations, n_dims, n_k, dim_a, k_a):  
    
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
            (rules_count_p[dim][k], rules_index_p[dim][k], rules_count_o[dim][k], rules_index_o[dim][k],
            rules_count_d[dim][k], rules_index_d[dim][k], rules_count_m[dim][k], rules_index_m[dim][k],
            rules_count_s[dim][k], rules_index_s[dim][k], rules_count_f[dim][k], rules_index_f[dim][k],
            rules_count_e[dim][k], rules_index_e[dim][k])\
            = _get_a_podmsfe_b_count(n_samples, all_shapelet_locations, dim_a, k_a, dim, k)

    return rules_count_p, rules_index_p, rules_count_o, rules_index_o, rules_count_d, rules_index_d, rules_count_m, rules_index_m, rules_count_s, rules_index_s, rules_count_f, rules_index_f, rules_count_e, rules_index_e

#Returns 7 5D arrays (one for each rule type) where all a>b possible rules are counted. The first and 
#second dimensions represent the [dim][k] of the antecedent, the third and fourth
#dimensions represent the [dim][k] of the subsequent, and the fifth dimension
#contains an array where the number of times a->b happens in each sample is 
@njit
def _get_all_as_podmsfe_bs_count(n_samples, all_shapelet_locations):
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
    
    rules_shs_p = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_shs_o = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_shs_d = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_shs_m = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_shs_s = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_shs_f = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    rules_shs_e = np.zeros((n_dims, n_k, n_dims, n_k, 4), dtype=np.int32)
    
    for dim in range(n_dims):
        for k in range(n_k):
            (rules_counts_p[dim][k], rules_indices_p[dim][k], rules_counts_o[dim][k], rules_indices_o[dim][k],
            rules_counts_d[dim][k], rules_indices_d[dim][k], rules_counts_m[dim][k], rules_indices_m[dim][k],
            rules_counts_s[dim][k], rules_indices_s[dim][k], rules_counts_f[dim][k], rules_indices_f[dim][k],
            rules_counts_e[dim][k], rules_indices_e[dim][k])\
            = _get_all_a_podmsfe_bs_count(n_samples, all_shapelet_locations, n_dims, n_k, dim, k)
            
            for dim_b in range(n_dims):
                for s in range(n_k):
                    rules_shs_p[dim][k][dim_b][s], rules_shs_o[dim][k][dim_b][s],\
                        rules_shs_d[dim][k][dim_b][s], rules_shs_m[dim][k][dim_b][s],\
                        rules_shs_s[dim][k][dim_b][s], rules_shs_f[dim][k][dim_b][s],\
                        rules_shs_e[dim][k][dim_b][s] = np.array([dim, k, dim_b, s], dtype=np.int32),\
                        np.array([dim, k, dim_b, s], dtype=np.int32),\
                        np.array([dim, k, dim_b, s], dtype=np.int32),\
                        np.array([dim, k, dim_b, s], dtype=np.int32),\
                        np.array([dim, k, dim_b, s], dtype=np.int32),\
                        np.array([dim, k, dim_b, s], dtype=np.int32),\
                        np.array([dim, k, dim_b, s], dtype=np.int32)
                
    return (rules_counts_p, rules_indices_p, rules_counts_o, rules_indices_o, rules_counts_d, rules_indices_d, rules_counts_m, rules_indices_m, rules_counts_s, rules_indices_s, rules_counts_f, rules_indices_f, rules_counts_e, rules_indices_e, rules_shs_p, rules_shs_o, rules_shs_d, rules_shs_m, rules_shs_s, rules_shs_f, rules_shs_e)



class RuleTransform(_PanelToTabularTransformer):
    def __init__(self,
          min_shapelet_length=3,
		  max_shapelet_length=np.inf,
          shapelets_lengths=None,
          occ_threshold=20,
          random_state=None,
          verbose=0,
          remove_self_overlapping=True,
          clustering_ratio=100,
          ):
         self.min_shapelet_length = min_shapelet_length
         self.max_shapelet_length = max_shapelet_length
         self.shapelets_lengths = shapelets_lengths
         self.occ_threshold = occ_threshold
         self.random_state = random_state
         self.verbose = verbose
         self.remove_self_overlapping = remove_self_overlapping
         self.clustering_ratio = clustering_ratio
         self.is_fitted_ = False
         super(RuleTransform, self).__init__()

    def fit(self, X, y=None):
        """A method to fit the rule transform to a specified X and y

        Parameters
        ----------
        X: pandas DataFrame
            The training input samples.
        y: array-like or list
            The class values for X

        Returns
        -------
        self : FullShapeletTransform
            This estimator
        """
        X, y = check_X_y(X, y, enforce_univariate=False, coerce_to_numpy=False)

        if (
            type(self) is ContractedRuleTransform
            and (self.shapelet_mining_contract <= 0 or self.rule_mining_contract <=0)
        ):
            raise ValueError("Error: time limit cannot be equal to or less than 0")

        self._n_ts = X.shape[0]
        self._n_dims = X.shape[1]
        self._ts_length = X.iloc[0,0].shape[0]

        ###CHECK THIS
        self._random_state = check_random_state(self.random_state)

        self._fit_transform_st(X, y)
        self._fit_rt(X, y)
        
        self.is_fitted_ = True

        return self


    def _fit_transform_st(self, X, y=None):
    	###CHECK THIS: it should work for 1D too
        #if self._n_dims == 1:
        if type(self) is ContractedRuleTransform:
            shapelet_mining_contract_per_dim = int(self.shapelet_mining_contract/self._n_dims)

			#If the time contract per dimensions is less than one minute, sample 
	        #time_contract_in_mins random dimensions and apply the ST to them
	        
	        #seed = 10
	        
            ###CHECK THIS: didn't work
            if shapelet_mining_contract_per_dim < 1:
	        	###CHECK THIS
	            #random.seed(seed)
	            dims = [random.randint(0, X.shape[1]-1) for p in range(0, int(shapelet_mining_contract_per_dim))]
	                
	            X = X.iloc[:, dims]
	            
	            #Spend one minute on each dimension
	            shapelet_mining_contract_per_dim = 1

            st = ContractedShapeletTransform(
    	            time_contract_in_mins=shapelet_mining_contract_per_dim,
    	            num_candidates_to_sample_per_case=10,
    	            min_shapelet_length=self.min_shapelet_length,
    	            max_shapelet_length=self.max_shapelet_length,
    	            verbose=self.verbose,
    	            predefined_ig_rejection_level=0.001,
    	        )

        else:
            st = ShapeletTransform(
                shapelets_lengths=self.shapelets_lengths,
                verbose=self.verbose,
                predefined_ig_rejection_level=0.001,
            )

        self._transformer = MultivariateTransformer(st)
        self._transformer.fit(X, y)
        self._transformer.transform(X)

    def _fit_rt(self, X, y=None):
        #Get the shapelets_distances from the transformer
        _shapelets_distances =  get_shapelets_distances(self._transformer)

        if len(_shapelets_distances) == 0:
            raise RuntimeError(
                "No shapelets were extracted that exceeded the "
                "minimum information gain threshold. Please retry with other "
                "data and/or ST parameter settings."
            )

        #No clustering
        if self.clustering_ratio == 100:
            self._all_shapelet_locations = self._get_all_shapelet_locations_scaled_threshold(_shapelets_distances, self._ts_length, self.occ_threshold/100.)
            del _shapelets_distances
        #Clustering shapelets
        else:
            #Get the shapelets from the transformer
            _shapelets =  get_shapelets(self._transformer)

            _clusters_distances = self._cluster_shapelets_and_format(_shapelets,_shapelets_distances,self.clustering_ratio)
            del _shapelets, _shapelets_distances
            self._all_shapelet_locations = self._get_all_cluster_locations_scaled_threshold(_clusters_distances, self._ts_length, self.occ_threshold/100.)
            del _clusters_distances

        if type(self) is ContractedRuleTransform:
            #Count the support of each rule
            (self._mined_rules, self._rules_counts_p, self._rules_indices_p, self._rules_counts_o,
            self._rules_indices_o, self._rules_counts_d, self._rules_indices_d,
            self._rules_counts_m, self._rules_indices_m, self._rules_counts_s, 
            self._rules_indices_s, self._rules_counts_f, self._rules_indices_f,
            self._rules_counts_e, self._rules_indices_e, self._rules_shs_p,
            self._rules_shs_o, self._rules_shs_d, self._rules_shs_m,
            self._rules_shs_s, self._rules_shs_f, self._rules_shs_e)\
            = _get_all_as_podmsfe_bs_count_contracted(self._n_ts, typed.List(self._all_shapelet_locations),
                                                      self.rule_mining_contract)
        
        else:
            (self._mined_rules, self._rules_counts_p, self._rules_indices_p, self._rules_counts_o,
            self._rules_indices_o, self._rules_counts_d, self._rules_indices_d,
            self._rules_counts_m, self._rules_indices_m, self._rules_counts_s, 
            self._rules_indices_s, self._rules_counts_f, self._rules_indices_f,
            self._rules_counts_e, self._rules_indices_e, self._rules_shs_p,
            self._rules_shs_o, self._rules_shs_d, self._rules_shs_m,
            self._rules_shs_s, self._rules_shs_f, self._rules_shs_e)\
            = _get_all_as_podmsfe_bs_count(self._n_ts, typed.List(self._all_shapelet_locations))

    def transform(self, X, y=None, test=False):
        if not self.is_fitted_:
            print('Need to call fit() first.')
            
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=False)

        if test:
            self._transformer.transform(X)
            #Get the test shapelets_distances from the transformer
            _shapelets_distances =  get_shapelets_distances(self._transformer)

            #No clustering
            if self.clustering_ratio == 100:
                self._all_shapelet_locations = self._get_all_shapelet_locations_scaled_threshold(_shapelets_distances, self._ts_length, self.occ_threshold/100.)
                del _shapelets_distances
            #Clustering shapelets
            else:
                #Get the shapelets from the transformer
                _shapelets =  get_shapelets(self._transformer)

                _clusters_distances = self._cluster_shapelets_and_format(_shapelets,_shapelets_distances,self.clustering_ratio,test=test)
                del _shapelets, _shapelets_distances
                self._all_shapelet_locations = self._get_all_cluster_locations_scaled_threshold(_clusters_distances, self._ts_length, self.occ_threshold/100.)
                del _clusters_distances

            if type(self) is ContractedRuleTransform:
                (_rules_counts_p, _rules_indices_p, _rules_counts_o, _rules_indices_o, _rules_counts_d, _rules_indices_d,
                _rules_counts_m, _rules_indices_m, _rules_counts_s, _rules_indices_s, _rules_counts_f, _rules_indices_f, _rules_counts_e, _rules_indices_e)\
                = _get_all_as_podmsfe_bs_count_contracted(self._n_ts, typed.List(self._all_shapelet_locations), rules_list=self._mined_rules, test=test)
            else:
                (_rules_counts_p, _rules_indices_p, _rules_counts_o, _rules_indices_o, _rules_counts_d, _rules_indices_d,
                _rules_counts_m, _rules_indices_m, _rules_counts_s, _rules_indices_s, _rules_counts_f, _rules_indices_f, 
                _rules_counts_e, _rules_indices_e, _, _, _, _, _, _, _)\
                = _get_all_as_podmsfe_bs_count(self._n_ts, typed.List(self._all_shapelet_locations))

            _all_rules_counts = self._transform(X, y, _rules_counts_p, _rules_counts_o, _rules_counts_d, _rules_counts_m, _rules_counts_s, _rules_counts_f, _rules_counts_e, test=test)
            del _rules_counts_p, _rules_counts_o, _rules_counts_d, _rules_counts_m, _rules_counts_s, _rules_counts_f, _rules_counts_e
        else:
             _all_rules_counts = self._transform(X, y)
             
        return _all_rules_counts

    def _transform(self, X, y=None, _rules_counts_p=None, _rules_counts_o=None, _rules_counts_d=None, _rules_counts_m=None, _rules_counts_s=None, _rules_counts_f=None, _rules_counts_e=None, test=False):
        if not test:
            _rules_counts_p = self._rules_counts_p
            _rules_counts_o = self._rules_counts_o
            _rules_counts_d = self._rules_counts_d
            _rules_counts_m = self._rules_counts_m
            _rules_counts_s = self._rules_counts_s
            _rules_counts_f = self._rules_counts_f
            _rules_counts_e = self._rules_counts_e
            
            _rules_indices_p = self._rules_indices_p
            _rules_indices_o = self._rules_indices_o
            _rules_indices_d = self._rules_indices_d
            _rules_indices_m = self._rules_indices_m
            _rules_indices_s = self._rules_indices_s
            _rules_indices_f = self._rules_indices_f
            _rules_indices_e = self._rules_indices_e
            
            _rules_shs_p = self._rules_shs_p
            _rules_shs_o = self._rules_shs_o
            _rules_shs_d = self._rules_shs_d
            _rules_shs_m = self._rules_shs_m
            _rules_shs_s = self._rules_shs_s
            _rules_shs_f = self._rules_shs_f
            _rules_shs_e = self._rules_shs_e
        
        _rules_counts_p = _rules_counts_p.reshape(-1,X.shape[0])
        _rules_counts_o = _rules_counts_o.reshape(-1,X.shape[0])
        _rules_counts_d = _rules_counts_d.reshape(-1,X.shape[0])
        _rules_counts_m = _rules_counts_m.reshape(-1,X.shape[0])
        _rules_counts_s = _rules_counts_s.reshape(-1,X.shape[0])
        _rules_counts_f = _rules_counts_f.reshape(-1,X.shape[0])
        _rules_counts_e = _rules_counts_e.reshape(-1,X.shape[0])
        
        _rules_indices_p = _rules_indices_p.reshape(-1,X.shape[0])
        _rules_indices_o = _rules_indices_o.reshape(-1,X.shape[0])
        _rules_indices_m = _rules_indices_d.reshape(-1,X.shape[0])
        _rules_indices_d = _rules_indices_m.reshape(-1,X.shape[0])
        _rules_indices_s = _rules_indices_s.reshape(-1,X.shape[0])
        _rules_indices_f = _rules_indices_f.reshape(-1,X.shape[0])
        _rules_indices_e = _rules_indices_e.reshape(-1,X.shape[0])
        
        if not test:
            _rules_shs_p = _rules_shs_p.reshape(-1,X.shape[0])
            _rules_shs_o = _rules_shs_o.reshape(-1,X.shape[0])
            _rules_shs_m = _rules_shs_m.reshape(-1,X.shape[0])
            _rules_shs_d = _rules_shs_d.reshape(-1,X.shape[0])
            _rules_shs_s = _rules_shs_s.reshape(-1,X.shape[0])
            _rules_shs_f = _rules_shs_f.reshape(-1,X.shape[0])
            _rules_shs_e = _rules_shs_e.reshape(-1,X.shape[0])
        
        #Concatenate the rules counts, indices, and shs (for the training set)
        _all_rules_counts = np.concatenate((_rules_counts_p, _rules_counts_o, _rules_counts_d,\
                                            _rules_counts_m, _rules_counts_s, _rules_counts_f,\
                                            _rules_counts_e), axis=0)
        _all_rules_indices = np.concatenate((_rules_indices_p, _rules_indices_o, _rules_indices_d,\
                                            _rules_indices_m, _rules_indices_s, _rules_indices_f,\
                                            _rules_indices_e), axis=0)
        if not test:
            _all_rules_shs = np.concatenate((_rules_shs_p, _rules_shs_o, _rules_shs_d,\
                                                _rules_shs_m, _rules_shs_s, _rules_shs_f,\
                                                _rules_shs_e), axis=0)

        #Indices of rules between inexistant shapelets (flagged -1 supports)
        _to_delete = np.where(np.all(_all_rules_counts==-1, axis=1))
        
        #Delete the -1 columns
        _all_rules_counts = np.delete(_all_rules_counts, _to_delete, axis=0)
        _all_rules_indices = np.delete(_all_rules_indices, _to_delete, axis=0)
        if not test:
            _all_rules_shs = np.delete(_all_rules_shs, _to_delete, axis=0)
                
        return _all_rules_counts.T, _all_rules_indices.T, _all_rules_shs.T

    #Returns the threshold used to select shapelet occurences based on a given percentage
    def _get_occurences_threshold(self, shapelets_distances, ts_length, percentage):
        #List to hold all distances values
        sds = []
        
        #Append the scaled distances
        for dim in shapelets_distances:
            for shapelet_distances in dim:
                #Compute the length of the shapelet
                shapelet_length = ts_length - shapelet_distances.shape[1] + 1
                for instance in shapelet_distances:
                    for distance in instance:
                        sds.append(distance/shapelet_length)
                    
        #Sort the distances ascendingly
        sds.sort()
        
         #Number of shapelet occurences to keep (per shapelet)
        n = int(percentage*len(sds)) 
        
        #Return the threshold distance to select the shapelet occurences to keep        
        return sds[n]

    #Given the shapelet_locations and shapelet_distances of one single shapelet, removes
    #the overlapping shapelet locations except the closest to the original
    def _remove_similar_locations(self, shapelet_locations, shapelet_distances):
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
    #the closest shapelets from the entire dataset 
    def _get_shapelet_locations_scaled_threshold(self, shapelet_distances, ts_length, threshold):
        #Compute the length of the shapelet
        shapelet_length = ts_length - shapelet_distances.shape[1] + 1
        
        #Get the indices of the n closest shapelets to the original shapelet
        s_indices = []
        for i in range(shapelet_distances.shape[0]):
            for j in range(shapelet_distances.shape[1]):
                #Compare to the threshold, scaled to shapelet length
                if shapelet_distances[i][j]/shapelet_length <= threshold:
                    s_indices.append(np.array([i,j]))
           
        if len(s_indices)>0:
            s_indices = np.asarray(s_indices) 
            
            #Create an array to store the locations of the closest n shapelets
            shapelet_locations = np.empty((s_indices.shape[0], s_indices.shape[1]+1), dtype=np.uint32)
            #Each shapelet is represented by (sample_index, start, end)
            for i in range(shapelet_locations.shape[0]):
                shapelet_locations[i] = np.append(s_indices[i], s_indices[i][1]+shapelet_length)
            
            #Remove overlapping shapelets and keep the closest one to th original shapelet
            shapelet_locations = self._remove_similar_locations(shapelet_locations, shapelet_distances)
        
            return shapelet_locations
        
        else:
            return np.array(np.array([[-1,-1,-1]]), dtype=np.uint32)

    #Get the locations of the closest shapelets for each shapelet across the 
    #entire dataset based on a chosen percentage
    def _get_all_shapelet_locations_scaled_threshold(self, shapelets_distances, ts_length, percentage):
        #Get the threshold to be used for selecting shapelet occurences
        threshold = self._get_occurences_threshold(shapelets_distances, ts_length, percentage)
        
        all_shapelet_locations = []
        for dim in shapelets_distances:
            dim_shapelet_locations = []
            for shapelet in dim:
                dim_shapelet_locations.append(self._get_shapelet_locations_scaled_threshold(shapelet, ts_length, threshold))
            all_shapelet_locations.append(dim_shapelet_locations)
            
        return all_shapelet_locations

    def _cluster_distance_dtw(self, shapelets, c1, c2, dim):
        distance = 0.0
        num_comparisons = 0

        for s1 in c1:
            for s2 in c2:
                distance += dtw(shapelets[s1].flatten(), shapelets[s2].flatten())
                num_comparisons += 1

        return distance/num_comparisons

    # Cluster similar shapelets of same lenght until the number of clusters at each
    # dimension equals n_dim
    def _cluster_shapelets(self, shapelets, n_c):
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
                            distances[i][j] = self._cluster_distance_dtw(shapelets[dim], clusters_shapelets[i], clusters_shapelets[j], dim)

                # Get the indices of the closest clusters
                ij = np.unravel_index(distances.argmin(), distances.shape)

                # Group the two clusters
                for sh in clusters_shapelets[ij[1]]:
                    clusters_shapelets[ij[0]].append(sh)
                del clusters_shapelets[ij[1]]
                
            all_clusters_shapelets.append(np.asarray(clusters_shapelets))

        return all_clusters_shapelets

    #Cluster shapelets based on a clustering ration and merge their shapelets_distances according 
    #to single shapelet's format
    def _cluster_shapelets_and_format(self, shapelets, shapelets_distances, cluster_ratio, test=False, all_clusters_shapelets=None):
        if not test:
            #List containing the desired number of clusters per dimension
            n_clusters = []
            
            for sd in shapelets_distances:
                n_shapelets = len(sd)
                n_clusters.append(int(n_shapelets*cluster_ratio/100))
    
            # if all(i < 10 for i in all_n_shapelets):
            #     print("Less than 10 shapelets per dimension. Not clustering.")
            #     return
            
            #Cluster
            self.all_clusters_shapelets = self._cluster_shapelets(shapelets, n_clusters)
        
        #List to hold the distances between each cluster-shapelet and its potential 
        #occurences, for all clusters at all dimensions
        clusters_distances = []
        
        #Merge shapelets_distances of clusters
        for dim in range(len(self.all_clusters_shapelets)):  
            #List to hold the distances between each cluster-shapelet and its potential 
            #occurences, for all clusters at this dimension
            dim_cluster_distances = []
            
            for c in range(len(self.all_clusters_shapelets[dim])):
                #List to hold the distances between this cluster-shapelet and its 
                #potential occurences
                cluster_distances = []
                
                for sh in self.all_clusters_shapelets[dim][c]:
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
                        
                dim_cluster_distances.append(np.asarray(cluster_distances))
                
            clusters_distances.append(np.asarray(dim_cluster_distances))
        
        # if test:
            # return clusters_distances
        # else:
        #     return self.all_clusters_shapelets, clusters_distances
        return clusters_distances

    #Given the cluster_distances matrix of a given cluster, get the locations of 
    #the closest shapelets from the entire dataset 
    def _get_cluster_locations_scaled_threshold(self, cluster_distances, ts_length, threshold):
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
            shapelet_locations = self._remove_similar_locations(shapelet_locations, new_sds)
        
            return shapelet_locations
        
        else:
            return np.array(np.array([[-1,-1,-1]]), dtype=np.uint32)

    #Get the locations of the closest shapelets for each cluster-shapelet across the 
    #entire dataset based on a chosen percentage
    def _get_all_cluster_locations_scaled_threshold(self, clusters_distances, ts_length, threshold):    
        all_cluster_locations = []
        for dim in clusters_distances:
            dim_cluster_locations = []
            for cluster in dim:
                dim_cluster_locations.append(self._get_cluster_locations_scaled_threshold(cluster, ts_length, threshold))
            all_cluster_locations.append(dim_cluster_locations)
            
        return all_cluster_locations

class ContractedRuleTransform(RuleTransform):
    def __init__(self,
                 min_shapelet_length=3,
                 max_shapelet_length=np.inf,
                 shapelets_lengths=None,
                 occ_threshold=20,
                 random_state=None,
                 verbose=0,
                 remove_self_overlapping=True,
                 clustering_ratio=100,
                 shapelet_mining_contract=None,
                 rule_mining_contract=None
                 ):

        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.shapelets_lengths = shapelets_lengths
        self.occ_threshold = occ_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.remove_self_overlapping = remove_self_overlapping
        self.clustering_ratio = clustering_ratio
        self.is_fitted_ = False
        self.shapelet_mining_contract = shapelet_mining_contract
        self.rule_mining_contract = rule_mining_contract

        super().__init__(
             min_shapelet_length,
             max_shapelet_length,
             shapelets_lengths,
             occ_threshold,
             random_state,
             verbose,
             remove_self_overlapping,
             clustering_ratio,
        )
