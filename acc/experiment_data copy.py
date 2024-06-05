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
        self.accuracy = []
        self.train_accuracy = []
        self.pool_accuracy = []
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
    
    def summarize_metric(self, df, col, auc, index):
        if auc:
            df[col] = df[col].apply(lambda x: (np.trapz(x, axis=1)/x.shape[1]).reshape(-1, 1))
        else:
            df[col] = df[col].apply(lambda x: x[:, index].reshape(-1, 1))
        return df
    
    def summarize_AL_procedure(self, df, auc=True, method="auc_max_ind", indices=[], threshold=1):
        if not auc and method == "auc_custom_ind" and len(indices) == 0:
            print("Need to specify indices for non-auc method with custom indices")
            return None
        df = df.copy()
        for metric in self.metrics:
            col = "mean_" + metric
            df[col] = df[metric].apply(lambda x: np.mean(x, axis=0)) # axis 0 (i.e., shape is (num_repeats, num_iterations))
            df['array_lengths'] = df[col].apply(lambda x: len(x))
            min_length = df['array_lengths'].min()
            max_length = df['array_lengths'].max()
            df['last_index_above_threshold'] = df[col].apply(lambda x: np.where(x > threshold)[0][0] if any(x > threshold) else len(x))
            max_index = df['last_index_above_threshold'].max()
            min_index = df['last_index_above_threshold'].min()

            if method == "batch_size":
                ind_step = 2000
                df = self.extend_dataframe(df, metric, max_length)
                for i, row in df.iterrows():
                    N = len(self.Y)
                    n_edges = (N*(N-1))/2
                    batch_size = row["batch_size"]
                    if batch_size < 1:
                        # should not be reached
                        continue
                    indices = []
                    for j in range(1, max_index):
                        if (j * batch_size) % ind_step == 0:
                            indices.append(j)
                        if (j*batch_size)/n_edges > 0.3:
                            break
                    indices = np.array(indices)
                    df.at[i, metric] = np.mean(df.at[i, metric][:, indices], axis=1).reshape(-1, 1)
            elif method == "auc_max_ind":
                df = self.extend_dataframe(df, metric, max_length)
                df = self.summarize_metric(df, metric, auc, max_length)
            elif method == "auc_max_thresh":
                df = self.extend_dataframe(df, metric, max_index)
                df = self.summarize_metric(df, metric, auc, max_index)
            elif method == "auc_min_ind":
                df = self.extend_dataframe(df, metric, min_length)
                df = self.summarize_metric(df, metric, auc, min_length)
            elif method == "auc_min_thresh":
                df = self.extend_dataframe(df, metric, min_index)
                df = self.summarize_metric(df, metric, auc, min_index)
            elif method == "auc_custom_ind":
                df = self.extend_dataframe(df, metric, max_length)
                if auc:
                    max_ind = np.max(indices)
                    df = self.extend_dataframe(df, metric, max_ind)
                    df = self.summarize_metric(df, metric, auc, max_ind)
                else:
                    df = self.extend_dataframe(df, metric, max_length)
                    df[metric] = df[metric].apply(lambda x: x[:, indices])
                    df[metric] = df[metric].apply(lambda x: np.mean(x, axis=1).reshape(-1, 1))
            else:
                raise ValueError("Invalid method")
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
        vary,
        options_in_file_name,
        auc=False, 
        summary_method="auc_max_ind", 
        indices=[], 
        threshold=[], 
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
        vary_options = {}

        for key, value in config.items():
            if key[0] == "_":
                continue
            if type(value) != list:
                value = [value]
            
            options_keys.append(key)
            if key not in compare and key not in vary:
                options_values.append(value)
            else:
                options_values.append([1111111111])

        for option in compare:
            compare_options[option] = config[option]

        if "x" not in vary:
            for option in vary:
                vary_options[option] = config[option]
            data = self.summarize_AL_procedure(
                data,
                auc=auc, 
                method=summary_method, 
                indices=indices, 
                threshold=threshold
            )
        
        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))

            for option in compare:
                exp_kwargs[option] = compare_options[option]

            if "x" not in vary:
                for option in vary:
                    exp_kwargs[option] = vary_options[option]

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
                df_filtered = df_filtered[df_filtered[vary[0]] < (min_length + 5)]

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

                if "x" in vary:
                    var = "total_queries"
                var = vary[0]
                ax = sns.lineplot(
                    x=vary[0],
                    y="y",
                    hue=df_filtered[hues].apply(tuple, axis=1),
                    #hue="acq_fn",
                    #hue_order=original_hue_order,
                    #hue_order=["info_gain_pairs_random", "info_gain_object", "info_gain_pairs", "entropy", "maxexp", "maxmin", "unif"],
                    errorbar=errorbar,
                    marker=".",
                    err_style=err_style,
                    data=df_filtered,
                    linestyle=linestyle,
                    err_kws=err_kws,
                    #palette=filtered_palette
                    #palette=palette
                )

                #plt.setp(ax.lines, markeredgecolor='none')  # Removes the border of the markers
                #plt.setp(ax.lines, alpha=0.7)  # Adjusts the transparency of the markers
                plt.setp(ax.lines, markeredgewidth=0.5)  # Adjusts the transparency of the markers
                #plt.setp(ax.lines, markersize=7)  # Adjusts the transparency of the markers
                if self.dataset == "ecoli":
                    mz = 8
                else:
                    mz = 10
                plt.setp(ax.lines, markersize=0)  # Adjusts the transparency of the markers
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))


                plt.ylabel(metric_map[metric])

                #labels = self.construct_x_ticks(ax, prop)
                #ws = 1-exp_kwargs["warm_start"]
                N = len(self.Y)
                n_edges = (N*(N-1))/2
                #N_pool = int(n_edges*ws)
                #rest = n_edges - N_pool
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
                ax.set_xticklabels(labels)

                plt.xlabel("Number of queries")
                #ax.legend(loc="best")
                ax.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.subplots_adjust(right=0.75)

                #legend = ax.get_legend()
                #if self.dataset != "synthetic":
                    #ax.get_legend().set_visible(False)

                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                #plt.savefig(file_path, bbox_extra_artists=(legend,), dpi=150, bbox_inches='tight')
                #plt.savefig(file_path, dpi=200, bbox_inches='tight')
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