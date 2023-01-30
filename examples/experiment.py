import os
import numpy as np
from sklearn.metrics import accuracy_score
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier

from utils import multivariate_contracted_st

#Path to the folder containing the original dataset
data_path = os.path.join(os.sep,'')

#Name of the folder where the program will save useful files for this specific
#configuration of the algorithm
name = 'rt_uea'

#load all datasets and perform ST, RT, and LS
for name in os.listdir(DATA_PATH):
    try:
    	if name in ('CharacterTrajectories', 'InsectWingbeat', 'JapaneseVowels', 'SpokenArabicDigits'):
    		continue

    	#Load the dataset
        print('Dataset:')
        print(name)
        
        X_train, y_train = load_from_tsfile_to_dataframe(
            os.path.join(DATA_PATH, name, name + "_TRAIN.ts")
        )
        X_test, y_test = load_from_tsfile_to_dataframe(
                os.path.join(DATA_PATH, name, name + "_TEST.ts")
            )

        #Perform ST for 6 hours and fit a random classifer on the shapelet space
        X_new, X_test_new = multivariate_contracted_st(X_train, y_train, X_test, 340, 20)
        
        clf = RandomForestClassifier(random_state=0, n_estimators=500)
        clf.fit(X_new, y_train)
        y_pred=clf.predict(X_test_new)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Accuracy on ST (6 hours): " + str(accuracy))

        #Get lengths of shapelets to mine
        min_length, max_length = get_shapelets_lengths_interval(X_train, y_train, total_time=20)
        
        rt = ContractedRuleTransform(
            shapelet_mining_contract=220,
            rule_mining_contract=120,
            min_shapelet_length=min_length,
            max_shapelet_length=max_length,
            verbose=0,
        )

        rt.fit(X_train.iloc y_train)
        all_rules_counts = rt.transform(X_train, test=False)
        all_rules_counts_test = rt.transform(X_test, test=True)

        #Indices of rules between inexistant shapelets (flagged -1 supports)
        to_delete = np.where(np.all(all_rules_counts==-1,axis=1))
        
        #Delete the -1 (fill) columns
        all_rules_counts = np.delete(all_rules_counts, to_delete, axis=0)
        all_rules_counts_test = np.delete(all_rules_counts_test, to_delete, axis=0)
                
        #Get fisher scores and sort list of indices
        scores = fisher_score.fisher_score(all_rules_counts, y_train)
        best_rules_indices = np.argsort(scores)[::-1]

        percentages = [100, 50, 20, 10, 5, 1, 0.1]  #the percentages of rules to keep during feature selection

        for i, percentage in enumerate(percentages):
            top_k = int(all_rules_counts.shape[1]*percentage/100)

            if top_k > 0:     
                best_rules_indices = best_rules_indices[:top_k]

                #Get the best rules
                best_rules = all_rules_counts[:,best_rules_indices]
                best_rules_test = all_rules_counts_test[:,best_rules_indices] 
                
                #Fit and Transform a Random Forest Classifier
                clf = RandomForestClassifier(random_state=0, n_estimators=500)
                clf.fit(best_rules, y_train)
                y_pred = clf.predict(best_rules_test)

                #Print the accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy on RT (4+2 hours) with " + str(percentage) + '(%) of the rule space: ' + str(accuracy))

except Exception as e:
        print(e)
        continue
