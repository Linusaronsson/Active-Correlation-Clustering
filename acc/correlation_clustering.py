import numpy as np
import sys
from scipy.special import softmax

def max_correlation(S, K, num_iterations):
    N = np.size(S, 0)

    best_objective = -np.finfo(np.float64).max 
    best_solution = np.zeros((N,), dtype=int)
        
    N_iter = 30

    for itr in range(num_iterations):
        current_solution = np.random.randint(0, K, size=N)

        # calc total cost
        current_objective = 0.0
        for i in range(N):
            for j in range(i):
                if current_solution[i] == current_solution[j]:
                    current_objective += S[i, j]

        old_objective = current_objective - 1

        while True:
            if (current_objective - old_objective) <= np.finfo(np.float64).eps or N_iter <= 0:
                break
            N_iter -= 1
            old_objective = current_objective
            order = np.random.permutation(N)
            
            for i in range(N):
                cur_ind = order[i]
                temp_objects = np.zeros((K,), dtype=float)
                
                for j in range(N):
                    if j != cur_ind:
                        temp_objects[current_solution[j]] += S[cur_ind, j]

                sep_Obj = temp_objects[current_solution[cur_ind]]
                temp_objects[current_solution[cur_ind]] = current_objective
                
                for k in range(K):
                    if k != current_solution[cur_ind]:
                        temp_objects[k] = current_objective - sep_Obj + temp_objects[k]
                                             
                temp_max = np.argmax(temp_objects)
                current_solution[cur_ind] = temp_max
                current_objective = temp_objects[temp_max]

        if itr == 0 or current_objective > best_objective:
            best_solution = np.array(current_solution)
            best_objective = current_objective
            
    return best_solution, best_objective

def max_correlation_dynamic_K(S, K, num_iterations):
    N = np.size(S, 0)
    #print("SIZE: ", N)
    K_dyn = K
    
    #print("NUM CLUSTERS: ", K_dyn)

    best_objective = -sys.float_info.max 
    best_solution = np.zeros((N,), dtype=int)

    N_iter = 30

    for itr in range(0,num_iterations):
        current_solution = np.zeros((N,), dtype=int) 
        
        for i in range(0,N):
            current_solution[i] = np.random.randint(0, K_dyn)
            
        # to gaurantee non-empty clusters
        temp_indices = np.random.choice(np.arange(0, N), K_dyn, replace=False)
        for k in range(0,K_dyn):
            current_solution[temp_indices[k]] = k

        current_objective = 0.0
        for k in range(0, K_dyn):
            inds = np.where(current_solution == k)[0]
            lower_triangle_indices = np.tril_indices(len(inds), -1) 
            current_objective += np.sum(S[np.ix_(inds, inds)][lower_triangle_indices])

        old_objective = current_objective - 1.0

        for _ in range(N_iter): 
            if (current_objective-old_objective) <= sys.float_info.epsilon:
                break
            old_objective = current_objective
            indices = np.arange(0, N)
            np.random.shuffle(indices)
            for i in indices:
                temp_objects = np.zeros(K_dyn)
                for k in range(0, K_dyn):
                    inds = np.where(current_solution == k)[0]
                    inds = inds[inds != i]
                    temp_objects[k] = np.sum(S[i, inds])

                if np.max(temp_objects) < 0.0:
                    # cerate a new cluster
                        current_objective = current_objective - temp_objects[current_solution[i]]
                        current_solution[i] = K_dyn
                        K_dyn = K_dyn + 1
                else:
                    sep_Obj = temp_objects[current_solution[i]]
                    temp_objects[current_solution[i]] = current_objective
                    for k in range(0,K_dyn):
                        if k != current_solution[i]:
                            temp_objects[k] = current_objective - sep_Obj + temp_objects[k]

                    temp_old_cluster = current_solution[i]
                    temp_max = np.argmax(temp_objects)
                    current_solution[i] = temp_max
                    current_objective = temp_objects[temp_max]
                        
                    # check the empy cluster, shinke if necessary
                    K_dyn_temp = len(np.unique(current_solution))
                    if K_dyn_temp < K_dyn:
                        for j in range(0,N):
                            if current_solution[j] > temp_old_cluster:
                                current_solution[j] = current_solution[j] - 1
                        K_dyn = K_dyn - 1

        if itr == 0 or current_objective > best_objective:
            best_solution = np.array(current_solution)
            best_objective = current_objective
            
    return best_solution, best_objective