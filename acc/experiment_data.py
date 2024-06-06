import os
import pickle
import math
import itertools
import json
import copy

import numpy as np
from hashlib import sha256
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

class ExperimentData:
    def __init__(self, Y, repeat_id, **kwargs):
        self.__dict__.update(kwargs)

        self.repeat_id = repeat_id
        self.experiment_params = []
        self.name = ""
        self.dataset_full_name = ""
        for key, value in kwargs.items():
            if key[0] == "_":
                continue
            if key.split("_")[0] == "dataset":
                self.dataset_full_name += str(value) + "_"
            self.name += str(value) + "_"
            self.experiment_params.append(key)
        self.name = self.name[:-1]
        self.name_repeat = self.name + "_" + str(repeat_id)
        self.hashed_name = sha256(self.name.encode("utf-8")).hexdigest()
        self.dataset_full_name = self.dataset_full_name[:-1]

        # ground truth clustering solution
        self.Y = Y

        # metrics
        self.rand = []
        self.ami = []
        self.v_measure = []
        self.num_clusters = []
        self.num_violations = []
        self.time = []
        self.time_select_batch = []
        self.time_update_clustering = []
        self.num_repeat_queries = []

    def is_equal_no_repeat(self, a2):
        params = set(self.experiment_params + a2.experiment_params)
        for param in params:
            if not hasattr(self, param) or not hasattr(a2, param):
                continue
            if getattr(self, param) != getattr(a2, param):
                return False
        return True

    def is_equal(self, a2):
        return self.is_equal_no_repeat(a2) and self.repeat_id == a2.repeat_id

class ExperimentReader:
    def __init__(self, metrics=["rand"]):
        self.metrics = metrics
        self.Y = None

    def get_metric_data(self, exp):
        data = {}
        for metric in self.metrics:
            for e in exp:
                if metric not in data:
                    data[metric] = [getattr(e, metric)]
                else:
                    data[metric].append(getattr(e, metric))
        return data

    def extend_list(self, data, max_size):
        if len(data) < max_size:
            data.extend([data[-1]] * (max_size - len(data)))
        #if len(data) > max_size:                
            #raise ValueError("Length of data larger than max size")
        return data

    def read_all_data(self, folder):
        data = pd.DataFrame()
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if name == "completed_experiments.txt":
                    continue
                with open(path + "/" + name, 'rb') as handle:
                    try:
                        exp = pickle.load(handle)
                    except EOFError:
                        print("EOF error in read data")
                        continue
                    dat = exp[0].__dict__
                    if self.Y is None:
                        self.Y = exp[0].Y
                    wanted_keys = exp[0].experiment_params
                    sub_dat = dict((k, dat[k]) for k in wanted_keys if k in dat)
                    metric_data = self.get_metric_data(exp)
                    for metric in self.metrics:
                        met_data = metric_data[metric]
                        max_size = 0
                        for mt in met_data:
                            if len(mt) > max_size:
                                max_size = len(mt)
                        for i in range(len(met_data)):
                            met_data[i] = self.extend_list(met_data[i], max_size)
                        sub_dat[metric] = np.array(met_data)
                    sub_dat = pd.DataFrame([sub_dat])
                    data = pd.concat([data, sub_dat], ignore_index=True)
        if len(data) == 0:
            print("no data found in folder")
            return None
        return data

    def flatten_dataframe(self, df, non_data_column_names, data_column_names):
        # List to store each DataFrame
        dataframes = []
        for data_column in data_column_names:
            # Temporarily drop non-required columns
            df_temp = df.drop(columns=[col for col in data_column_names if col != data_column])
            
            # Expand each array into its own DataFrame and merge back with the original DataFrame
            for i in range(df_temp.shape[0]):
                array = df_temp.loc[i, data_column]
                df_expanded = pd.DataFrame(array).unstack().reset_index()
                df_expanded.columns = ['x', 'x2', 'y']
                for col in non_data_column_names:
                    df_expanded[col] = [df_temp.loc[i, col]] * len(df_expanded)
                df_expanded['metric'] = data_column
                dataframes.append(df_expanded)

        # Concatenate all dataframes
        df_flattened = pd.concat(dataframes, ignore_index=True)
        return df_flattened
    
    def extend_list_all(self, data, max_size):
        new_dat = data.tolist()
        for i in range(len(data)):
            new_dat[i] = self.extend_list(new_dat[i], max_size)
        return np.array(new_dat)

    def extend_dataframe(self, df, col, index):
        df[col] = df[col].apply(lambda x: self.extend_list_all(x, index))
        df[col] = df[col].apply(lambda x: x[:, :index+1])
        #df[col] = df[col][:, :index+1]
        return df
    
    def filter_dataframe(self, df, conditions):
        mask = pd.Series(True, index=df.index)
        for column, values in conditions.items():
            if type(values) != list:
                values = [values]
            mask = mask & df[column].isin(values)
        return df[mask]

    def generate_AL_curves(
        self,
        data,
        save_location,
        categorize,
        compare,
        options_in_file_name,
        err_style="band",
        marker=None,
        markersize=6,
        capsize=6,
        linestyle="solid",
        prop=True,
        **config):

        config = copy.deepcopy(config)

        options_keys = []
        options_values = []
        compare_options = {}

        for key, value in config.items():
            if key[0] == "_":
                continue
            if type(value) != list:
                value = [value]
            
            options_keys.append(key)
            if key not in compare:
                options_values.append(value)
            else:
                options_values.append([1111111111])

        for option in compare:
            compare_options[option] = config[option]

        
        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))

            for option in compare:
                exp_kwargs[option] = compare_options[option]

            for metric in self.metrics:
                df_filtered = self.filter_dataframe(data, exp_kwargs).reset_index()
                #if metric in ["time_select_batch", "time_update_clustering", "time", "num_violations", "num_repeat_queries"]:
                    #continue
                col = "mean_" + metric
                df_filtered[col] = df_filtered[metric].apply(lambda x: np.mean(x, axis=0))
                df_filtered['array_lengths'] = df_filtered[col].apply(lambda x: len(x))
                max_length = df_filtered['array_lengths'].max()
                min_length = df_filtered['array_lengths'].min()
                df_filtered = self.extend_dataframe(df_filtered, metric, max_length)


                data_column_names = [metric]
                non_data_column_names = list(set(data.columns) - set(data_column_names))
                if df_filtered.shape[0] == 0:
                    print("No data for these options @@@@@@@")
                    continue
                df_filtered = self.flatten_dataframe(df_filtered, non_data_column_names, data_column_names)

                path = save_location + "/" + metric + "/"
                for option in categorize:
                    path += str(exp_kwargs[option]) + "/" 
                fig_path = Path(path)
                fig_path.mkdir(parents=True, exist_ok=True)

                file_name = metric + "_"
                for option in options_in_file_name:
                    file_name += str(exp_kwargs[option]) + "_"
                file_name = file_name[:-1] + ".png"

                file_path = path + file_name
                self.dataset = exp_kwargs["dataset_name"] 
                self.batch_size = exp_kwargs["batch_size"] 

                hues = list(compare_options.keys())
                sns.set_theme()
                sns.set_style("white")
                SMALL_SIZE = 16
                MEDIUM_SIZE = 18
                BIGGER_SIZE = 18

                plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
                plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
                plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('legend', fontsize=18)    # legend fontsize
                plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
                plt.rc('figure', dpi=150)
                plt.rc('figure', figsize=(6, 4))

                # Customize gridlines
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)

                # Set the border color and width
                ax = plt.gca()  # Get current axis
                for _, spine in ax.spines.items():
                    spine.set_color('black')
                    spine.set_linewidth(1)

                if err_style == "bars":
                    err_kws = {
                        "capsize": capsize,
                        "marker": marker,
                        "markersize": markersize,
                    }
                else:
                    err_kws = {}

                errorbar = ("sd", 1)
                cut_threshold = 400
                df_filtered = df_filtered[df_filtered["x"] < cut_threshold]
                #df_filtered = df_filtered[df_filtered["x"] < (min_length + 5)]

                metric_map = {
                    "ami": "AMI", "rand": "ARI", "time": "Time (s)", "num_violations": "Num. violations",
                    "time_select_batch": "Time (s)", "time_update_clustering": "Time (s)",
                    "num_repeat_queries": "Num. pairs re-queried", "accuracy": "Accuracy", 
                    "v_measure": "V-measure", "train_accuracy": "Train accuracy", "pool_accuracy": "Pool accuracy"
                }

                acq_fn_map = {
                    "maxexp": "Maxexp", "maxmin": "Maxmin", "uncert": "Uncertainty",
                    "freq": "Frequency", "unif": "Uniform", "nCOBRAS": "nCOBRAS",
                    "COBRAS": "COBRAS", "QECC": "QECC", "info_gain_object": "EIG-O", "info_gain_pairs": "EIG-P",
                    "cluster_incon": "IMU-C", "entropy": "Entropy", "info_gain_pairs_random": "JEIG"
                }

                ax = sns.lineplot(
                    x="x",
                    y="y",
                    hue=df_filtered[hues].apply(tuple, axis=1),
                    #hue="acq_fn",
                    #hue_order=[],
                    errorbar=errorbar,
                    marker=".",
                    err_style=err_style,
                    data=df_filtered,
                    linestyle=linestyle,
                    err_kws=err_kws,
                    #palette=palette
                )

                #plt.setp(ax.lines, markeredgecolor='none')  # Removes the border of the markers
                #plt.setp(ax.lines, alpha=0.7)  # Adjusts the transparency of the markers
                plt.setp(ax.lines, markeredgewidth=0)  # Adjusts the transparency of the markers
                #plt.setp(ax.lines, markersize=7)  # Adjusts the transparency of the markers
                plt.setp(ax.lines, markersize=0)  # Adjusts the transparency of the markers
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                plt.ylabel(metric_map[metric])
                N = len(self.Y)
                n_edges = (N*(N-1))/2
                rest = 0
                if self.batch_size < 1:
                    batch_size = math.ceil(n_edges * self.batch_size)
                else:
                    batch_size = self.batch_size
                labels = []
                for item in ax.get_xticks():
                    if prop:
                        labels.append(round((int(item)*batch_size+rest)/n_edges, 2))
                    else:
                        lab = int(int(item)*batch_size+rest)
                        labels.append(lab)
                ax.set_xticks(ax.get_xticks().tolist()[1:])
                ax.set_xticklabels(labels[1:])

                if prop:
                    plt.xlabel("Number of queries")
                else:
                    plt.xlabel("Number of queries divided by number of edges")
                #ax.legend(loc="best")
                ax.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.subplots_adjust(right=0.75)

                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.clf()

    def generate_experiments(self, folder, options_to_keep, start_index=1, **config):
        options_keys = []
        options_values = []
        i = start_index
        fig_path = Path(folder)
        fig_path.mkdir(parents=True, exist_ok=True)

        for key, value in config.items():
            if type(value) != list:
                value = [value]
            if key not in options_to_keep:
                options_values.append(value)
            else:
                options_values.append([1])
            options_keys.append(key)

        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))
            for key, value in exp_kwargs.items():
                if key in options_to_keep:
                    exp_kwargs[key] = config[key]

            with open(folder + "/experiment" + str(i) + ".json", "w") as fp:
                json.dump(exp_kwargs, fp, indent=4)
            i += 1
        return i