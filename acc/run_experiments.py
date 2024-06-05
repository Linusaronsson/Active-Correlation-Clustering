from acc.active_clustering import ActiveClustering

import os
import multiprocessing as mp
import argparse
import json
import pickle
import itertools

import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from pathlib import Path

def get_dataset(**options):
    dataset = options["dataset_name"]
    seed = options["seed"]
    if dataset == "synthetic":
        class_balance = options["dataset_class_balance"]
        n_clusters = options["dataset_n_clusters"]
        n_samples = options["dataset_n_samples"]
        class_sep = options["dataset_class_sep"]
        y_flip = options["dataset_y_flip"]
        n_features = options["dataset_n_features"]
        if class_balance == None:
            weights = None
        else:
            prop = (1-class_balance)/(n_clusters-1)
            weights = [class_balance]
            weights += [prop]*(n_clusters-1)
        X, Y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_clusters,
            n_clusters_per_class=1,
            weights=weights,
            flip_y=y_flip,
            class_sep=class_sep,
            hypercube=True,
            shift=0.0,
            scale=1.0,
            shuffle=True,
            random_state=seed)
    elif dataset == "20newsgroups":
        X = np.load("datasets/20newsgroups_data/X.npy")
        Y = np.load("datasets/20newsgroups_data/Y.npy")
    elif dataset == "cifar10":
        X = np.load("datasets/cifar10_data/X.npy")
        Y = np.load("datasets/cifar10_data/Y.npy")
    elif dataset == "mnist":
        X = np.load("datasets/mnist_data/X.npy")
        Y = np.load("datasets/mnist_data/Y.npy")
    elif dataset == "breast_cancer":
        X = np.load("datasets/breast_cancer_data/X.npy")
        Y = np.load("datasets/breast_cancer_data/Y.npy")
    elif dataset == "cardiotocography":
        X = np.load("datasets/cardiotocography_data/X.npy")
        Y = np.load("datasets/cardiotocography_data/Y.npy")
    elif dataset == "ecoli":
        X = np.load("datasets/ecoli_data/X.npy")
        Y = np.load("datasets/ecoli_data/Y.npy")
    elif dataset == "forest_type_mapping":
        X = np.load("datasets/ForestTypeMapping_data/X.npy")
        Y = np.load("datasets/ForestTypeMapping_data/Y.npy")
    elif dataset == "mushrooms":
        X = np.load("datasets/mushrooms_data/X.npy")
        Y = np.load("datasets/mushrooms_data/Y.npy")
    elif dataset == "user_knowledge":
        X = np.load("datasets/user_knowledge_data/X.npy")
        Y = np.load("datasets/user_knowledge_data/Y.npy")
    elif dataset == "yeast":
        X = np.load("datasets/yeast_data/X.npy")
        Y = np.load("datasets/yeast_data/Y.npy")
    else:
        raise ValueError("INVALID DATASET")
    return X, Y

def gather_results(result_queue, path):
    num_completed = 0
    try:
        while True:
            ac_data = result_queue.get(block=True, timeout=None)
            if ac_data is None:
                return
            experiment_path = path + ac_data.dataset_full_name
            data_path = experiment_path + "/" + ac_data.hashed_name
            if not os.path.exists(data_path):
                Path(experiment_path).mkdir(parents=True, exist_ok=True)
                if os.path.exists(experiment_path):
                    with open(data_path, 'wb') as handle:
                        pickle.dump([ac_data], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        num_completed += 1
            else:
                with open(data_path, 'rb') as handle:
                    exp = pickle.load(handle)
                exp.append(ac_data)
                with open(data_path, 'wb') as handle:
                    pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    num_completed += 1

            completed_experiments_path = path + "completed_experiments.txt"
            with open(completed_experiments_path, "a") as file:
                file.write(ac_data.name_repeat + "\n")

    except EOFError as e:
        print("EOF ERROR GATHER RESULTS !! !! @ ")
        print(e)
        print("--------------")

def run_experiment(experiment_queue, result_queue, worker_id, path):
    try:
        while True:
            ac = experiment_queue.get(block=True, timeout=None)
            if ac is None:
                return
            ac_data = ac.ac_data
            print("#### Worker ID: {} ####".format(worker_id))
            already_completed = False
            completed_experiments_path = path + "completed_experiments.txt"
            with open(completed_experiments_path) as file:
                for line in file:
                    if line.strip() == ac_data.name_repeat:
                        already_completed = True
                        break
            if already_completed:
                print(ac.ac_data.name_repeat + " already completed")
                continue
            else:
                print(ac.ac_data.name_repeat + " running")
            print("#### ####")
            ac_data = ac.run_AL_procedure()
            result_queue.put(ac_data)
    except EOFError as e:
        print("EOF ERROR RUN_EXPERIMENT !! !! !! !! !!")
        print(e)
        print("--------------")
        return

def get_keys_from_options(config):
    options_keys = []
    options_values = []
    for key1, value1 in config.items():
        for key2, value2 in value1.items():
            if type(value2) != list:
                value2 = [value2]
            options_keys.append(key2)
            options_values.append(value2)
    return options_keys, options_values

def run_experiments(config):
    manager = mp.Manager()
    experiment_queue = manager.Queue() 
    result_queue = manager.Queue() 

    path = "experiment_results_local/" 
    path += config["_experiment_name"] + "/"
    exp_results = Path(path)
    completed_experiments_path = path + "completed_experiments.txt"
    completed_exps = Path(completed_experiments_path)
    exp_results.mkdir(parents=True, exist_ok=True)
    completed_exps.touch(exist_ok=True)

    options_keys = []
    options_values = []
    for key, value in config.items():
        if type(value) != list:
            value = [value]
        options_keys.append(key)
        options_values.append(value)

    saved_datasets = {}
    for repeat_id in range(config["_num_repeats"]):
        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))
            dataset_name = ""
            for key, value in exp_kwargs.items():
                if key.split("_")[0] == "dataset":
                    dataset_name += str(value)

            X, Y, = get_dataset(**exp_kwargs)
            seed = exp_kwargs["seed"]
            if dataset_name not in saved_datasets and exp_kwargs["sim_init_type"] == "kmeans":
                saved_datasets[dataset_name] = {}
                k = len(np.unique(Y))
                kmeans = KMeans(n_clusters=k, random_state=seed).fit(X)
                saved_datasets[dataset_name]["initial_labels"] = kmeans.labels_

            if exp_kwargs["sim_init_type"] == "kmeans":
                initial_labels = saved_datasets[dataset_name]["initial_labels"]
            else:
                initial_labels = None

            ac = ActiveClustering(X, Y, repeat_id, initial_labels, **exp_kwargs)

            experiment_queue.put(ac)

            if config["_overwrite"] and os.path.exists(completed_experiments_path):
                ac_data = ac.ac_data
                experiment_path = path + ac_data.dataset_full_name
                data_path = experiment_path + "/" + ac_data.hashed_name
                with open(completed_experiments_path, "r") as f:
                    lines = f.readlines()
                with open(completed_experiments_path, "w") as file:
                    for line in lines:
                        if line.strip() != ac_data.name_repeat:
                            file.write(line)
                if os.path.exists(data_path):
                    os.remove(data_path)


    processes = []
    for worker in range(config["_n_workers"]):
        process = mp.Process(target=run_experiment, args=(experiment_queue, result_queue, worker, path), daemon=False)
        process.start()
        processes.append(process)
        experiment_queue.put(None)
    gather_process = mp.Process(target=gather_results, args=(result_queue, path), daemon=False)
    gather_process.start()

    #gather_process.join()
    #for process in processes:
    #   process.terminate()
    #result_queue.put(None) 

    for process in processes:
        process.join()
    result_queue.put(None) 
    gather_process.join()

def read_config_file(filename):
    if filename.split('.')[-1] not in ['json']:
        raise IOError('Only json type are supported now!')
	
    if not os.path.exists(filename):
        raise FileNotFoundError('Config file does not exist!')
        
    with open(filename, 'r') as f:
        config = json.load(f)
    return config	

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, help="Config")

    args = parser.parse_args()
    mp.set_start_method("spawn")

    if ".json" in args.config:
        config = read_config_file(args.config)
        run_experiments(config)
    else:
        for file in os.listdir(args.config):
            filename = os.fsdecode(file)
            config = read_config_file(args.config + "/" + filename)
            run_experiments(config)

if __name__ == "__main__":
    main()