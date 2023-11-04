
import os   
import ast
import sys


import matplotlib.pyplot as plt
from matplotlib import image as mpimg

from typing import Optional, List
from dataclasses import dataclass
from itertools import repeat
from omegaconf import OmegaConf, MISSING

import torch as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import csv
import h5py
import pickle
from scipy.signal import savgol_filter
from batchnorm.color_logging import ColorLogger, make_timestamp
from batchnorm.incremental_hdf5 import IncrementalHDF5

import tueplots as tplt
from tueplots import cycler, bundles
from tueplots.constants.color import palettes

"""

This file provides various functions to gather and plot the results from the BatchNorm experiments, to ease comparison.


functions:
    - performance_across_ablations: compare training loss and accuracy of the models across all ablation modifications for one position
    - compare_ablation_performance: compare the loss of the models across all ablation modifications for two positions respectively
    - compare_multiple_lr: compare training loss and accuracy of the models across different learning rates
    - paramMean_perLayer: plot the mean of the beta and gamma parameters for each layer across training time
    - bn_params_3Dsurface: plot the beta and gamma parameters for each layer of a model as a contpour plot and a 3D surface plot
    - compare_3Dsurface_betas: plot 3D surface of the beta parameters for each layer across modifications of a model
    - compare_contour_betas: plot contour plot of the beta parameters for each layer across modifications of a model 
"""


#################################################################################################
# HELPERS
#################################################################################################

## Process log path

def read_log(path):
    log_path = [p for p in os.listdir(path) if p.endswith("log")][0]
    log_data = pd.DataFrame(pd.read_json(os.path.join(path, log_path), lines=True).iloc[:, 1].values.tolist())
    log_data = {k: pd.DataFrame(entry[0] for entry in log_data[log_data.iloc[:, 0].str.contains(k)].iloc[:, 1:].values)
                for k in ["PARAMETERS", "SETTINGS", "EVAL ROUND", "BATCH"]}
    return log_data


def get_file_path(directory, extension):
    files = [p for p in os.listdir(directory) if p.endswith(extension)]
    return os.path.join(directory, files[0]) if files else None


def read_and_process_log(path):
    log = read_log(path)
    
    return {
        'train_loss': log["EVAL ROUND"]["train_loss"],
        'train_acc': log["EVAL ROUND"]["train_acc"],
        'global_step': log["EVAL ROUND"]["global_step"],
        'position': log["PARAMETERS"]["POSITION"][0],
        'shapes': log["SETTINGS"]['shapes'],
        'num_epochs': log["SETTINGS"]["num_epochs"][0],
        'seed': log["PARAMETERS"]["RANDOM_SEED"][0],
        'problem': log["PARAMETERS"]["PROBLEM"][0],
        'ablation_mode': log["PARAMETERS"]["ABLATION"][0],
        'lr': log["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0],
        'batch_size': log["SETTINGS"]["batch_size"][0]
    }


## Create plot directory

def create_plot_dir(seed, problem, num_epochs, lr, name):
    plot_dir = os.path.join(
    "batchnorm/output/plots", f"{seed}__{problem}__ep_{num_epochs}", "Bias_Init_00", f"{name}", f"lr_{lr}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir



## Process data of affine transformation parameters

def get_mean_values(data_files, shapes):
    len_datafile = IncrementalHDF5.get_num_elements(data_files)
    mean_values = [[] for _ in range(len(shapes))]
    steps = []

    for idx in range(len_datafile):
        data, metadata = IncrementalHDF5.get_element(data_files, idx)
        step = ast.literal_eval(metadata)["step"]
        steps.append(step)

        start = 0
        for i, layer in enumerate(shapes):
            end = start + layer[0]
            mean_values[i].append(np.mean(data[start:end]))
            start = end
    steps = np.stack(steps)

    return mean_values, steps

def get_parameter_values(data_files, shapes):
    values = [[] for _ in range(len(shapes))]
    steps = []

    with h5py.File(data_files, "r") as f:
        len_datafile = f['data'].shape[0]
        for idx in range(len_datafile):
            start = 0
            for i, layer in enumerate(shapes):
                end = start + layer[0]
                values[i] = f['data'][start:end]
                start = end
    return values

## Helper for plotting functions

def plot_loss_curve(ax, step, loss1, loss2, loss_fullBN, loss_Vanilla, position1, position2, title):
    ax.plot(step, loss_Vanilla, label=f'vanilla', linestyle="dashed", color='palevioletred')
    ax.plot(step, loss_fullBN, label=f'1,1,1,1,1 $_\\beta, _\\gamma$', linestyle="dashed", color='purple')
    ax.plot(step, loss1, label=f'{position1} $_\\beta, _\\gamma$')
    ax.plot(step, loss2, label=f'{position2} $_\\beta, _\\gamma$')
    ax.set_title(title)
    ax.set_ylabel('train loss')
    ax.set_xlabel('training step')
    ax.set_ylim([0.25, 2.5])
    ax.set_xlim([0, max(step)])
    ax.legend()

def smooth(data, window_length=7, polyorder=2):
    return savgol_filter(data, window_length, polyorder)


#################################################################################################
# Compare training performance across all modifications
#################################################################################################
def performance_across_ablations(paths, smooth_curve=False):
    """
    Compares the performance of models based on different ablation modes and visualizes the results.

    Args:
        paths (list): List of paths to the logs of the training for each model modification.
        smooth (bool): Whether to smooth the loss curves or not.
    Returns:
        None. The function will save and display the loss and accuracy plots.
    """
    txt_logger = ColorLogger(f"[{os.path.basename(sys.argv[0])}]")
    txt_logger.info("PARAMETERS")


    # read log into dataframe and gather basic stats
    #keys = ["PARAMETERS", "SETTINGS", "EVAL ROUND", "BATCH"]
    #logs = [read_and_process_log(path) for path in paths]

    # Read data of Vanilla model
    log_Vanilla = read_log(paths[0])

    Vanilla_train_step = log_Vanilla["BATCH"]["global_step"]
    Vanilla_test_step = log_Vanilla["EVAL ROUND"]["global_step"]
    Vanilla_train_loss = log_Vanilla["EVAL ROUND"]["train_loss"]          #["EVAL ROUND"]["train_loss"]
    Vanilla_train_accuracy = log_Vanilla["EVAL ROUND"]["train_acc"]

    # Read data of Original BatchNorm
    log_BN_trained = read_log(paths[1])
    position = log_BN_trained["PARAMETERS"]["POSITION"][0]

    originalBN_train_step = log_BN_trained["BATCH"]["global_step"]
    originalBN_test_step = log_BN_trained["EVAL ROUND"]["global_step"]
    originalBN_train_loss = log_BN_trained["EVAL ROUND"]["train_loss"]
    originalBN_train_accuracy = log_BN_trained["EVAL ROUND"]["train_acc"]

    # Read data of Fixed Parameters
    log_nonAdapt = read_log(paths[2])
    nonAdapt_train_loss = log_nonAdapt["EVAL ROUND"]["train_loss"]
    nonAdapt_train_accuracy = log_nonAdapt["EVAL ROUND"]["train_acc"]

    # Read data of Adaptive Beta
    log_fixedGamma = read_log(paths[3])
    fixedGamma_train_loss = log_fixedGamma["EVAL ROUND"]["train_loss"]
    fixedGamma_train_accuracy = log_fixedGamma["EVAL ROUND"]["train_acc"]

    # Read data of Adaptive Gamma
    log_fixedBeta = read_log(paths[4])
    fixedBeta_train_loss = log_fixedBeta["EVAL ROUND"]["train_loss"]
    fixedBeta_train_accuracy = log_fixedBeta["EVAL ROUND"]["train_acc"]


    # data to create directory to save plots
    seed = log_Vanilla["PARAMETERS"]["RANDOM_SEED"][0]
    problem = log_Vanilla["PARAMETERS"]["PROBLEM"][0]
    num_epochs = log_Vanilla["SETTINGS"]["num_epochs"][0]
    bs = log_Vanilla["SETTINGS"]["batch_size"][0]
    lr = log_Vanilla["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0]
    # round lr two 4 digits
    lr_rounded = round(lr, 4)   

    stats_to_smooth = [
        Vanilla_train_loss,
        originalBN_train_loss,
        nonAdapt_train_loss,
        fixedGamma_train_loss,
        fixedBeta_train_loss,
        Vanilla_train_accuracy,
        originalBN_train_accuracy,
        nonAdapt_train_accuracy,
        fixedGamma_train_accuracy,
        fixedBeta_train_accuracy
        ]

    if smooth_curve:
        for stat in stats_to_smooth:
            stat = smooth(stat)

    # Create plot directory
    plot_dir = os.path.join(
        "batchnorm/output/plots",
        f"{seed}__{problem}__ep_{num_epochs}", f"batchsize_{bs}")
    os.makedirs(plot_dir, exist_ok=True)

    with plt.rc_context({**bundles.aistats2022(), **tplt.axes.lines()}):
        plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12 
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        axes[0].plot(Vanilla_test_step, Vanilla_train_loss, label="Vanilla", color="palevioletred")
        axes[0].plot(originalBN_test_step, originalBN_train_loss, label="BN$_\\gamma, _\\beta$", color="purple")
        axes[0].plot(originalBN_test_step, nonAdapt_train_loss, label="BN")
        axes[0].plot(originalBN_test_step, fixedGamma_train_loss, label="BN$_\\beta$")#, linestyle="dashed")
        axes[0].plot(originalBN_test_step, fixedBeta_train_loss, label="BN$_\\gamma$")#, linestyle="dashed")

        axes[0].set_ylabel('loss')
        axes[0].set_xlabel('training step')
        axes[0].set_xlim([100, max(originalBN_test_step)])
        axes[0].set_ylim([0, 3.5])

        axes[1].plot(Vanilla_test_step, Vanilla_train_accuracy, label="Vanilla", color="palevioletred")
        axes[1].plot(originalBN_test_step, originalBN_train_accuracy, label="BN$_\\gamma, _\\beta$", color="purple")
        axes[1].plot(originalBN_test_step, nonAdapt_train_accuracy, label="BN")
        axes[1].plot(originalBN_test_step, fixedGamma_train_accuracy, label="BN$_\\beta$")#, linestyle="dashed")
        axes[1].plot(originalBN_test_step, fixedBeta_train_accuracy, label="BN$_\\gamma$")#, linestyle="dashed")

        axes[1].set_ylabel('accuracy')
        axes[1].set_xlabel('training step')
        axes[1].set_xlim([100, max(originalBN_test_step)])
        axes[1].set_ylim([0.05, 1])
        
        axes[0].legend()
        axes[1].legend()
        # add grid
        axes[0].grid()
        axes[1].grid()
        #fig.suptitle(f"Comparison of losses and accuracy between Model with and without BatchNorm, with lr tuned for Vanilla model \n (seed: {seed}, problem: {problem}, lr: {lr}, bs: {bs}) ")
        #plt.show()

        if smooth_curve:
            fig.savefig(os.path.join(plot_dir, f"{seed}_ep_{num_epochs}_{position}_lr_{lr_rounded}_Smooth_LossAndAccuracy_AblationStudy.png"), dpi=350)
        else:
            fig.savefig(os.path.join(plot_dir, f"{seed}_ep_{num_epochs}_{position}_lr_{lr_rounded}_LossAndAccuracy_AblationStudy.png"), dpi=350)
        plt.close(fig)



#################################################################################################
# Usage Example
#
# model_paths = [f"batchnorm/output/4842821__cifar100_3c3d_bn__{x}/Lr_0.16579130972807002/Ep_350/{y}" for x, y 
#                in zip(["0,0,0,0,0", *["0,0,1,1,1"] * 4],
#                 ["trained_bn", "trained_bn", "fixed_bn", "fixed_gamma", "fixed_beta"])]
#
# performance_across_ablations(model_paths, smooth_curve=False)
#################################################################################################


def compare_ablation_performance(paths):
    """
    Compares the training loss of two models across all modifications
    Creates a figure of two plots, one for each model

    Args:
        paths (list): List of paths to the logs of the training for each model modification.

    Returns:
        None. The function will save and display the comparison plots.

    Example path list:
        paths_ablations_1layer = [
            "batchnorm/output/4842821__cifar100_3c3d_bn__0,0,0,0,0/Lr_0.16579130972807002/Ep_350/trained_bn",
            "batchnorm/output/4842821__cifar100_3c3d_bn__1,1,1,1,1/Lr_0.16579130972807002/Ep_350/trained_bn",
            "batchnorm/output/4842821__cifar100_3c3d_bn__1,0,0,0,0/Lr_0.16579130972807002/Ep_350/trained_bn",
            "batchnorm/output/4842821__cifar100_3c3d_bn__0,0,0,0,1/Lr_0.16579130972807002/Ep_350/trained_bn", 
            "batchnorm/output/4842821__cifar100_3c3d_bn__1,0,0,0,0/Lr_0.16579130972807002/Ep_350/fixed_bn", 
            "batchnorm/output/4842821__cifar100_3c3d_bn__0,0,0,0,1/Lr_0.16579130972807002/Ep_350/fixed_bn",
            "batchnorm/output/4842821__cifar100_3c3d_bn__1,0,0,0,0/Lr_0.16579130972807002/Ep_350/fixed_beta",
            "batchnorm/output/4842821__cifar100_3c3d_bn__0,0,0,0,1/Lr_0.16579130972807002/Ep_350/fixed_beta", 
            "batchnorm/output/4842821__cifar100_3c3d_bn__1,0,0,0,0/Lr_0.16579130972807002/Ep_350/fixed_gamma", 
            "batchnorm/output/4842821__cifar100_3c3d_bn__0,0,0,0,1/Lr_0.16579130972807002/Ep_350/fixed_gamma",
            ]
        compare_ablation_performance(paths_ablations_1layer)
    """


    logs = [read_and_process_log(path) for path in paths]

    # Plot parameters
    max_step = max(logs[0]['global_step'])
    seed = logs[0]['seed']
    problem = logs[0]['problem']
    lr = logs[0]['lr']
    num_epochs = logs[0]['num_epochs']
    batch_size = logs[0]['batch_size']
    baseline_position = logs[1]['position']
    
    name = "Compare_Ablations_Performance"
    ablation_labels = {
    'trained_bn': '$_\\beta, _\\gamma$',
    'fixed_bn': '$_-$',
    'fixed_gamma': '$_\\beta$',
    'fixed_beta': '$_\\gamma$'
    }

    plot_dir = create_plot_dir(seed, problem, num_epochs, lr, name)

    # Create figure
    with plt.rc_context({**bundles.aistats2022(), **tplt.axes.lines()}):
        plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
            
        fig, axs = plt.subplots(2, 1, figsize=(15, 8))


        # Divide logs for the subplots: Vanilla and FullBN always go to both, the rest are allocated to each subplot
        logs1 = [logs[0], logs[1]] + [logs[i] for i in range(2, len(logs), 2)]
        logs2 = [logs[0], logs[1]] + [logs[i] for i in range(3, len(logs), 2)]

        for i, log_set in enumerate([logs1, logs2]):
            for j, log in enumerate(log_set):
                linestyle = "dashed" if j < 2 else "solid"  # different line style for Vanilla and FullBN
                color = 'palevioletred' if j == 0 else 'purple' if j == 1 else None  # different colors for Vanilla and FullBN
                label = (
                    "vanilla" if j == 0 
                    else f'{baseline_position}$_\\beta, _\\gamma$' if j == 1 
                    else f'{log_set[2]["position"]} {ablation_labels.get(log["ablation_mode"])}')  # set label based on model
                axs[i].plot(log['global_step'], log['train_loss'], label=label, linestyle=linestyle, color=color)
            axs[i].set_title(f'Position {log_set[2]["position"]}')  # skip Vanilla and FullBN for position title
            axs[i].set_ylabel('train loss')
            axs[i].set_xlabel('training step')
            axs[i].set_ylim([0.25, 3.5])
            axs[i].set_yscale('log')
            axs[i].set_xlim([0, max_step])
            axs[i].grid()
            axs[i].legend()

        fig.suptitle(f'Performance comparison for each position of BatchNorm layers \n (seed {seed}, problem {problem}, lr {lr}, batch size {batch_size})', fontsize=16) # ({logs1[2]["position"]}) and ({logs2[2]["position"]})  \n (seed {seed}, problem {problem}, lr {lr})', fontsize=16)
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(plot_dir, f"{seed}_ep_{num_epochs}_Loss_comparison_{logs1[2]['position']}_and_{logs2[2]['position']}.svg"), dpi=350)
        plt.close(fig)


#################################################################################################
# Plot function for learning rate comparison
#################################################################################################

def compare_multiple_lr(paths):

    with plt.rc_context({**bundles.aistats2022(family="serif"), **tplt.axes.lines()}):
        tplt.axes.lines()
        plt.rcParams.update(tplt.axes.tick_direction(x="inout", y="in"))
        plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
        plt.rcParams.update(tplt.figsizes.neurips2021(nrows=2, ncols=3))
        plt.rcParams.update(tplt.fontsizes.neurips2021())

        fig, axs = plt.subplots(1,2, figsize=(10, 5))
        lrs = []
        
        for path in paths:
            # Read the log file
            log = read_log(path)
            
            # Extract necessary information
            train_loss = log["EVAL ROUND"]["train_loss"]
            train_acc = log["EVAL ROUND"]["train_acc"]
            global_step = log["EVAL ROUND"]["global_step"]
            
            position = log["PARAMETERS"]["POSITION"][0]
            num_epochs = log["SETTINGS"]["num_epochs"][0]
            seed = log["PARAMETERS"]["RANDOM_SEED"][0]
            problem = log["PARAMETERS"]["PROBLEM"][0]
            ablation_mode = log["PARAMETERS"]["ABLATION"][0]
            batch_size = log["SETTINGS"]["batch_size"][0]

            
            lr = log["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0]
            lrs.append(lr)

            # Update max step if necessary
            max_step = global_step.max()
            
            # Plot loss and accuracy
            axs[0].plot(global_step, train_loss, label=f'lr={lr}')
            axs[1].plot(global_step, train_acc, label=f'lr={lr}')
            
        # Create directory to store plots
        name = "Compare_LearningRates"
        plot_dir = os.path.join(
        "batchnorm/output/plots", f"{seed}__{problem}__ep_{num_epochs}", f"{name}")
        os.makedirs(plot_dir, exist_ok=True)
        plot_dir = create_plot_dir(seed, problem, num_epochs, lrs[0], name)
            
        # Set common properties for the plots
        axs[0].set(xlabel='global Step', ylabel='train loss')
        axs[0].set_xlim([0, max_step])
        axs[0].set_ylim([0, 1.2])
        axs[0].grid()
        axs[0].legend()

        axs[1].set(xlabel='global Step', ylabel='train accuracy')
        axs[1].set_xlim([0, max_step])
        axs[1].set_ylim([0.2, 1])

        axs[1].grid()
        axs[1].legend()
        fig.suptitle(f"Comparison of learning rates for position ({position})  \n (seed {seed}, problem {problem}, batch size {batch_size})")

        # Save the figures
        fig.savefig(os.path.join(plot_dir, f"{seed}_ep_{num_epochs}_LearningRate_Comparison_BNPosition_{position}.svg"), dpi=350)
        plt.show()
        

#################################################################################################
# Plot functions for the beta and gamma parameters in BatchNorm
#################################################################################################


def paramMean_perLayer(Vanilla_path, fullBN_path, paths):

    """
    Input Path structure:
        Vanilla_path: results from Vanilla model
        fullBN_path: BN model with fully trained BN layers
        paths: paths to modifications of the model with BatchNorm

    Return: 
        Creates a plot for each param and each layer
    """

    # Gather data from log file for the plots
    log_Vanilla = read_log(Vanilla_path)
    shapes_Vanilla = log_Vanilla["SETTINGS"]["shapes"][0]

    log_fullBN = read_log(fullBN_path)
    position_fullBN = log_fullBN["PARAMETERS"]["POSITION"][0]
    shapes_fullBN = log_fullBN["SETTINGS"]["shapes"][0]

    log = read_log(paths[0])
    shapes = log["SETTINGS"]["shapes"][0]
    position = log["PARAMETERS"]["POSITION"][0]
    seed = log["PARAMETERS"]["RANDOM_SEED"][0]
    num_epochs = log["SETTINGS"]["num_epochs"][0]
    problem = log["PARAMETERS"]["PROBLEM"][0]
    lr = log["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0]
    name = "Compare_BNparams_per_layer"
    batch_size = log["SETTINGS"]["batch_size"][0] 
    ablation_labels = {
    'trained_bn': '$_\\beta, _\\gamma$',
    'fixed_bn': '$_-$',
    'fixed_gamma': '$_\\beta$',
    'fixed_beta': '$_\\gamma$'
    }

    plot_dir = create_plot_dir(seed, problem, num_epochs, lr, name)

    # Gather params for Vanilla and FullBN models for comparison
    path_Vanilla_bias = get_file_path(Vanilla_path, "bias.h5")
    path_fullBN_beta = get_file_path(fullBN_path, "beta.h5")
    path_fullBN_gamma = get_file_path(fullBN_path, "gamma.h5")

    bias = h5py.File(path_Vanilla_bias, "r") if path_Vanilla_bias else None
    fullBN_beta = h5py.File(path_fullBN_beta, "r") if path_fullBN_beta else None
    fullBN_gamma = h5py.File(path_fullBN_gamma, "r") if path_fullBN_gamma else None

    bias_means, _ = get_mean_values(bias, shapes_Vanilla)
    fullBN_beta_means, _ = get_mean_values(fullBN_beta, shapes_fullBN)
    fullBN_gamma_means, _ = get_mean_values(fullBN_gamma, shapes_fullBN)

    # Create figure
    with plt.rc_context({**bundles.aistats2022(family="serif"), **tplt.axes.lines()}):
        tplt.axes.lines()
        plt.rcParams.update(tplt.axes.tick_direction(x="inout", y="in"))
        plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
        plt.rcParams.update(tplt.figsizes.neurips2021(nrows=2, ncols=3))
        plt.rcParams.update(tplt.fontsizes.neurips2021())
            
        fig, axes = plt.subplots(len(shapes), 2, figsize=(12, 4*len(shapes)))

        # Adjust for the case of a single row of subplots
        if len(shapes) == 1:
            axes = np.array([axes]) 

        for path in paths:

            log_temp = read_log(path)
            ablation_mode = log_temp["PARAMETERS"]["ABLATION"][0]

            path_beta = get_file_path(path, "beta.h5")
            path_gamma = get_file_path(path, "gamma.h5")

            beta = h5py.File(path_beta, "r") if path_beta else None
            gamma = h5py.File(path_gamma, "r") if path_gamma else None

            len_beta = IncrementalHDF5.get_num_elements(beta)
            steps = []
        
            for idx in range(len_beta):
                _, metadata = IncrementalHDF5.get_element(beta, idx)
                step = ast.literal_eval(metadata)["step"]
                steps.append(step)
            steps = np.stack(steps)

            if beta and gamma:
                beta_means, _ = get_mean_values(beta, shapes)
                gamma_means, _ = get_mean_values(gamma, shapes)
            

            position_str = ''.join(str(i) for i in position)
            layer_position = [i+1 for i, val in enumerate(position_str.split(',')) if val == '1']       
            
            for i, pos in enumerate(layer_position):
                axes[i][0].plot(steps, bias_means[pos-1], label=f'vanilla', linestyle="dashed", color='palevioletred')
                axes[i][0].plot(steps, fullBN_beta_means[pos-1], label=f'{position_fullBN} {ablation_labels.get(log["ablation_mode"])}', linestyle="dashed", color='purple')
                axes[i][1].plot(steps, fullBN_gamma_means[pos-1], label=f'{position_fullBN} {ablation_labels.get(log["ablation_mode"])}', linestyle="dashed", color='purple')
                # Set legend
                axes[i][0].legend()
                axes[i][1].legend()

            for i in range(len(beta_means)):
                if ablation_mode == "fixed_beta":
                    axes[i][0].plot(steps, beta_means[i], label=f'{position} {ablation_labels.get(log["ablation_mode"])}', linestyle="dotted", color='blue')
                    axes[i][1].plot(steps, gamma_means[i], label=f'{position} {ablation_labels.get(log["ablation_mode"])}')  
    
                elif ablation_mode == "fixed_gamma":
                    axes[i][0].plot(steps, beta_means[i], label=f'{position} {ablation_labels.get(log["ablation_mode"])}' )  
                    axes[i][1].plot(steps, gamma_means[i], label=f'{position} {ablation_labels.get(log["ablation_mode"])}', linestyle="dotted", color='darkgreen')  
                else: 
                    axes[i][0].plot(steps, beta_means[i], label=f'{position} {ablation_labels.get(log["ablation_mode"])}')  
                    axes[i][1].plot(steps, gamma_means[i], label=f'{position} {ablation_labels.get(log["ablation_mode"])}')  
                axes[i][0].set_ylabel(f'beta  mean')
                axes[i][1].set_ylabel(f'gamma mean')
                axes[i][0].set_xlabel('training step')
                axes[i][1].set_xlabel('training step')
                axes[i][0].set_xlim([0, max(steps)])
                axes[i][1].set_xlim([0, max(steps)])
                axes[i][0].set_ylim([-0.8,1.8])
                axes[i][1].set_ylim([0.2,2.8])
                axes[i][0].set_title(f"Layer {layer_position[i]} - Beta params over training time")
                axes[i][1].set_title(f"Layer {layer_position[i]} - Gamma params over training time")
            
                # Set legend
                axes[i][0].legend()
                axes[i][1].legend()
                
        beta.close()
        gamma.close()

        fig.suptitle(f"Comparison of beta and gamma for all ablation modes for BN norm layers positions: ({position})  \n (seed {seed}, problem {problem}, lr {lr}), batch size: {batch_size}")
        fig.savefig(os.path.join(plot_dir, f"{seed}_Comparison_BNparams_perLayer_{position}.png"), dpi=350)
        plt.show()


#################################################################################################
# Plot 3D surface of gamma and beta params over training time for each layer
#################################################################################################

def bn_params_3Dsurface(sort_by_converged, affine_param, path):
    """
    Args:
    sort_by_converged (boolean): if True, sort dimensions by converged value
    affine_param (string): 'beta' or 'gamma' or 'bias'(Vanilla model)
    path (string): path to model with BatchNorm layers

    Returns:
    Contour and 3D plots the of beta/gamma values of one model, for each layer across training time 

    Usage example:
        paths_AffineParams_gamma = [
            "batchnorm/output/4842821__cifar100_allcnnc_bn__1,0,0,0,0,0,0,0,0/Lr_0.16579130972807002/Ep_350/trained_bn",
            "batchnorm/output/4842821__cifar100_allcnnc_bn__1,0,0,0,0,0,0,0,0/Lr_0.16579130972807002/Ep_350/fixed_beta"
        ]

        for i, path in enumerate(paths_AffineParams_gamma):
            input_path = path
            bn_params_3Dsurface(True, "gamma", input_path)

    """

    # Get data from log and h5 files
    log_path = read_log(path)
    shapes = log_path["SETTINGS"]["shapes"][0]
    position = log_path["PARAMETERS"]["POSITION"][0]
    ablation = log_path["PARAMETERS"]["ABLATION"][0]

    seed = log_path["PARAMETERS"]["RANDOM_SEED"][0]
    num_epochs = log_path["SETTINGS"]["num_epochs"][0]
    problem = log_path["PARAMETERS"]["PROBLEM"][0]
    lr = log_path["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0]
    batch_size = log_path["SETTINGS"]["batch_size"][0]

    position_str = ''.join(str(i) for i in position)
    layer_position = [i+1 for i, val in enumerate(position_str.split(',')) if val == '1']

    if affine_param == "beta":
        path_params = get_file_path(path, "beta.h5")
    elif affine_param == "gamma":
        path_params = get_file_path(path, "gamma.h5")
    params = get_parameter_values(path_params, shapes)
    params_transposed = [param.T for param in params]

    # directory to store plots
    name = "ParameterActivity"
    plot_dir = create_plot_dir(seed, problem, num_epochs, lr, name)
    

    num_cols = max(1,len(shapes))

    ablation_labels = {
    'trained_bn': '$_\\beta, _\\gamma$',
    'fixed_bn': '$_-$',
    'fixed_gamma': '$_\\beta$',
    'fixed_beta': '$_\\gamma$'
    }
    # VScales to use if colormap should be consistent
    # Compute min and max values for each layer across all plots
    param_min_values = None
    param_max_values = None
    max_abs_values = None
    max_dev_from_one = None

    for params in params_transposed:
        if param_min_values is None:
            param_min_values = np.min(params)
        else:
            param_min_values = min(param_min_values, np.min(params))
        if param_max_values is None:
            param_max_values = np.max(params)
        else:
            param_max_values = max(param_max_values, np.max(params))
        if max_abs_values is None:
            max_abs_values = np.max(np.abs(params))
        else:
            max_abs_values = max(max_abs_values, np.max(np.abs(params)))
        if affine_param == "gamma":
            if max_dev_from_one is None:
                max_dev_from_one = np.max(np.abs(params - 1))
            else:
                max_dev_from_one = max(max_dev_from_one, np.max(np.abs(params - 1)))

    
    with plt.rc_context({**bundles.aistats2022(), **tplt.axes.lines()}):
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12 
        # Create Contour Plot
        if len(shapes) <= 5:
            fig1 = plt.figure(figsize=(2.5 * len(shapes), 2.5))
            num_rows = 1
            num_cols = len(shapes)
        elif len(shapes) > 5 and len(shapes) <= 8:
            fig1 = plt.figure(figsize=(2.5 * 5, 2.5 * 2))  # 2 rows
            num_rows = 2
            num_cols = 4  # At max 4 columns, as mentioned
        else:
            fig1 = plt.figure(figsize=(2.5 * 5, 2.5 * 3))  # 3 rows
            num_rows = 3
            num_cols = 3  # At max 4 columns, as mentioned

        for idx, layer in enumerate(shapes):
            # Determine current row and column for the subplot
            row_idx = idx // num_cols
            col_idx = idx % num_cols
            # Create a meshgrid for the neurons and training steps
            neurons = np.arange(layer[0])
            steps = np.arange(0, params_transposed[idx].shape[0] * 5, 5)#[:200] ###############################################################################################################################################################
            X, Y = np.meshgrid(neurons, steps)
            Z = params_transposed[idx]
            # sort dimensions by converged value
            if sort_by_converged:
                Z = Z[:,Z[-1].argsort()]

            ax1 = fig1.add_subplot(num_rows, num_cols, idx+1)
            if affine_param == "beta":
                c = ax1.contourf(
                    X, Y, Z, 
                    levels=np.linspace(param_min_values, param_max_values, 100), 
                    cmap='bwr', 
                    vmin=-max_abs_values, 
                    vmax=max_abs_values
                    )
            elif affine_param == "gamma":
                c = ax1.contourf(
                    X, Y, Z, 
                    levels=np.linspace(param_min_values, param_max_values, 100), 
                    cmap='bwr', 
                    vmin= 1 - max_abs_values, 
                    vmax= 1 + max_abs_values
                    )
            ax1.set_xlabel('dimension')
            ax1.set_ylabel('training step')
            ax1.set_title(f'Layer {layer_position[idx]}')
            if len(shapes) <= 5 and idx == len(shapes) - 1:
                fig1.colorbar(c)
            if len(shapes) > 5 and idx == 3:
                fig1.colorbar(c, ax=fig1.axes[-1])

        #fig2.suptitle(f"Comparison of betas across Batch Norm layers:{position} \n (seed {seed}, problem {problem}, lr {lr}), batch size {batch_size}")#, y=0.9)
        if sort_by_converged:
            plt.savefig(os.path.join(plot_dir, f"{seed}_{affine_param}_Activity_{position}_{ablation}_contourP_sorted.png"), dpi=350)
        else:
            plt.savefig(os.path.join(plot_dir, f"{seed}_{affine_param}_Activity_{position}_{ablation}_contourP.png"), dpi=350)
        #plt.show()
        #plt.close()

        # Create 3D Plot
        if len(shapes) <= 5:
            fig2 = plt.figure(figsize=(3*(max(1,len(shapes))), 3))
            num_rows = 1
            num_cols = len(shapes)
        else:
            fig2 = plt.figure(figsize=(3 * 5, 3 * 2))
            num_rows = 2
            num_cols = 4  


        for idx, layer in enumerate(shapes):
            # Create a meshgrid for the neurons and training steps
            neurons = np.arange(layer[0])
            steps = np.arange(0, params_transposed[idx].shape[0] * 5, 5)
            X, Y = np.meshgrid(neurons, steps)
            Z = params_transposed[idx]                    
            # sort dimensions by converged value
            if sort_by_converged:
                Z = Z[:,Z[-1].argsort()]

            ax2 = fig2.add_subplot(num_rows, num_cols, idx+1, projection='3d', proj_type='ortho')
            if affine_param == "beta":
                ax2.plot_surface(X, Y, Z, cstride=1, rstride=5, cmap='bwr', vmin=-max_abs_values, vmax=max_abs_values) 
            elif affine_param == "gamma":
                ax2.plot_surface(X, Y, Z, cstride=1, rstride=5, cmap='bwr', vmin= 1 - max_dev_from_one, vmax= 1 + max_dev_from_one)
            ax2.view_init(elev=30, azim=-45)
            ax2.set_box_aspect(None, zoom=0.9)
            ax2.set_zlim(param_min_values, param_max_values)
            ax2.set_xlabel('dimension')
            ax2.set_ylabel('training step')
            ax2.set_title(f'Layer {layer_position[idx]}')
            
        if sort_by_converged:
            plt.savefig(os.path.join(plot_dir, f"{seed}_{affine_param}_Activity_{position}_{ablation}_3D_sorted.png"), dpi=350)
        else:
            plt.savefig(os.path.join(plot_dir, f"{seed}_{affine_param}_Activity_{position}_{ablation}_3D.png"), dpi=350)
        #fig.suptitle(f"Comparison of betas across Batch Norm layers: ({position})  \n (seed {seed}, problem {problem}, lr {lr}), batch size {batch_size}")#, y=0.9)
        #plt.show()
        plt.close()
        



def compare_3Dsurface_betas(sort_by_converged, Vanilla_path, fullBN_path, trained_bn_path, fixed_gamma_path):
    """
    Args:
    sort_by_converged (boolean): if True, sort dimensions by converged value
    Vanilla_path (string): path to model without any BatchNorm layers
    fullBN_path (string): path to model with all BatchNorm layers
    trained_bn_path (string): path to model with BatchNorm layers at a specified position and ablation_mode is trained_bn
    fixed_gamma_path (string): path to model with BatchNorm layers at a specified position and ablation mode is fixed_gamma (because we plot betas here)

    Returns:
    None. 3D plot of the beta values for each layer across training time for Vanilla, fullBN, trained_bn and fixed_gamma models
    """

    # Preprocess data for plots
    log_Vanilla = read_log(Vanilla_path)
    shapes_Vanilla = log_Vanilla["SETTINGS"]["shapes"][0]

    path_bias = get_file_path(Vanilla_path, "bias.h5")
    biases = get_parameter_values(path_bias, shapes_Vanilla)
    bias_values_transposed = [bias.T for bias in biases]

    log_fullBN = read_log(fullBN_path)
    shapes_fullBN = log_fullBN["SETTINGS"]["shapes"][0]
    fullBN_position = log_fullBN["PARAMETERS"]["POSITION"][0]
    fullBN_ablation = log_fullBN["PARAMETERS"]["ABLATION"][0]

    path_full_beta = get_file_path(fullBN_path, "beta.h5")
    full_betas = get_parameter_values(path_full_beta, shapes_fullBN)
    full_beta_values_transposed = [beta.T for beta in full_betas]

    log_trained = read_log(trained_bn_path)
    shapes = log_trained["SETTINGS"]["shapes"][0]
    position = log_trained["PARAMETERS"]["POSITION"][0]
    seed = log_trained["PARAMETERS"]["RANDOM_SEED"][0]
    num_epochs = log_trained["SETTINGS"]["num_epochs"][0]
    problem = log_trained["PARAMETERS"]["PROBLEM"][0]
    lr = log_trained["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0]
    batch_size = log_trained["SETTINGS"]["batch_size"][0]
    ablation_trained = log_trained["PARAMETERS"]["ABLATION"][0]

    position_str = ''.join(str(i) for i in position)
    layer_position = [i+1 for i, val in enumerate(position_str.split(',')) if val == '1']       

    path_trained_beta = get_file_path(trained_bn_path, "beta.h5")
    trained_beta = get_parameter_values(path_trained_beta, shapes)
    trained_beta_values_transposed = [beta.T for beta in trained_beta]  

    log_fixedG = read_log(fixed_gamma_path)
    ablation_fixedG = log_fixedG["PARAMETERS"]["ABLATION"][0]
    path_fixedG_beta = get_file_path(fixed_gamma_path, "beta.h5")
    fixedG_beta = get_parameter_values(path_fixedG_beta, shapes)
    fixedG_beta_values_transposed = [beta.T for beta in fixedG_beta]

    # Create data lists for plotting
    beta_full_list = [bias_values_transposed, full_beta_values_transposed]
    beta_position_list = [trained_beta_values_transposed, fixedG_beta_values_transposed]
    ablation_modes = [ablation_trained, ablation_fixedG]

    # Create output dir
    name = "ParameterActivity_Comparison"
    plot_dir = create_plot_dir(seed, problem, num_epochs, lr, name)

    num_cols = len(beta_full_list) + len(beta_position_list)
    num_rows = max(1,len(shapes))

    ablation_labels = {
    'trained_bn': '$_\\beta, _\\gamma$',
    'fixed_bn': '$_-$',
    'fixed_gamma': '$_\\beta$',
    'fixed_beta': '$_\\gamma$'
    }
    # VScales to use if colormap should be consistent
    # Compute min and max values for each layer across all plots
    layer_min_values = {}
    layer_max_values = {}
    max_abs_values = {}

    # For beta_full_list
    for params in beta_full_list:
        for i, pos in enumerate(layer_position):
            Z = params[pos-1]
            if pos not in layer_min_values:
                layer_min_values[pos] = np.min(Z)
            else:
                layer_min_values[pos] = min(layer_min_values[pos], np.min(Z))
                
            if pos not in layer_max_values:
                layer_max_values[pos] = np.max(Z)
            else:
                layer_max_values[pos] = max(layer_max_values[pos], np.max(Z))
            if pos not in max_abs_values:
                max_abs_values[pos] = np.max(np.abs(Z))
            else:
                max_abs_values[pos] = max(max_abs_values[pos], np.max(np.abs(Z)))

    # For beta_position_list
    for params in beta_position_list:
        for i, layer in enumerate(shapes):
            Z = params[i]
            pos = layer_position[i]
            layer_min_values[pos] = min(layer_min_values[pos], np.min(Z))
            layer_max_values[pos] = max(layer_max_values[pos], np.max(Z))
            max_abs_values[pos] = max(max_abs_values[pos], np.max(np.abs(Z)))


    # Create 3D plots using matplotlib
    with plt.rc_context({**bundles.aistats2022(family="serif"), **tplt.axes.lines()}):
        tplt.axes.lines()
        plt.rcParams.update(tplt.axes.tick_direction(x="inout", y="in"))
        plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
        plt.rcParams.update(tplt.figsizes.neurips2021(nrows=2, ncols=3))
        plt.rcParams.update(tplt.fontsizes.neurips2021())
            
        fig = plt.figure(figsize=(15, 4*max(1,len(shapes))))

        # Plotting for beta_full_list
        for col, params in enumerate(beta_full_list):
            for i, pos in enumerate(layer_position):
                # Create a meshgrid for the neurons and training steps
                neurons = np.arange(shapes[i][0])
                steps = np.arange(0, params[pos-1].shape[0] * 5, 5)#[:200] ###############################################################################################################################################################
                X, Y = np.meshgrid(neurons, steps)
                Z = params[pos-1]
                 # sort dimensions by converged value
                if sort_by_converged:
                    Z = Z[:,Z[-1].argsort()]

                #Z = Z[:200] ###############################################################################################################################################################

                ax_idx = col + i * num_cols + 1 if num_rows > 1 else col + 1
                ax = fig.add_subplot(num_rows, num_cols, ax_idx, projection='3d', proj_type='ortho')
                ax.plot_surface(X, Y, Z, cstride=1, rstride=5, cmap='bwr', vmin=-max_abs_values[pos], vmax=max_abs_values[pos])
                ax.view_init(elev=30, azim=-45)
                ax.set_box_aspect(None, zoom=0.9)
                ax.set_zlim(layer_min_values[pos], layer_max_values[pos])
                ax.set_xlabel('dimension')
                ax.set_ylabel('training step')
                ax.set_title(f'Vanilla Biases (Layer {pos})' if col == 0 else f'{fullBN_position}{ablation_labels.get(fullBN_ablation)} Betas (Layer{pos})')
                
        # Plotting for beta_position_list
        offset = len(beta_full_list)
        for col, params in enumerate(beta_position_list):
            for row, layer in enumerate(shapes):
                # Create a meshgrid for the neurons and training steps
                neurons = np.arange(layer[0])
                steps = np.arange(0, params[row].shape[0] * 5, 5)#[:200] ###############################################################################################################################################################
                X, Y = np.meshgrid(neurons, steps)
                Z = params[row]                    
                # sort dimensions by converged value
                if sort_by_converged:
                    Z = Z[:,Z[-1].argsort()]
                #Z = Z[:200] ###############################################################################################################################################################

                ax_idx = offset + col + row * num_cols + 1 if num_rows > 1 else offset + col + 1
                ax = fig.add_subplot(num_rows, num_cols, ax_idx, projection='3d', proj_type='ortho')
                ax.plot_surface(X, Y, Z, cstride=1, rstride=5, cmap='bwr', vmin=-max_abs_values[layer_position[row]], vmax=max_abs_values[layer_position[row]]) 
                ax.view_init(elev=30, azim=-45)
                ax.set_box_aspect(None, zoom=0.9)
                ax.set_zlim(layer_min_values[layer_position[row]], layer_max_values[layer_position[row]])
                ax.set_xlabel('dimension')
                ax.set_ylabel('training step')
                ax.set_title(f'{position}{ablation_labels.get(ablation_modes[col])} Betas (Layer {layer_position[row]})')
                
        #fig.suptitle(f"Comparison of betas across Batch Norm layers: ({position})  \n (seed {seed}, problem {problem}, lr {lr}), batch size {batch_size}")#, y=0.9)
        plt.savefig(os.path.join(plot_dir, f"{seed}_BetaActivity_Comparison_{position}_3D.png"), dpi=350)
        

        # Create contour plots
        fig2 = plt.figure(figsize=(15, 3*max(1,len(shapes))))
        # Plotting for beta_full_list
        for col, params in enumerate(beta_full_list):
            for i, pos in enumerate(layer_position):
                # Create a meshgrid for the neurons and training steps
                neurons = np.arange(shapes[i][0])
                steps = np.arange(0, params[pos-1].shape[0] * 5, 5)
                X, Y = np.meshgrid(neurons, steps)
                Z = params[pos-1]
                # sort dimensions by converged value
                if sort_by_converged:
                    Z = Z[:,Z[-1].argsort()]

                ax_idx = col + i * num_cols + 1 if num_rows > 1 else col + 1
                ax2 = fig2.add_subplot(num_rows, num_cols, ax_idx)
                c = ax2.contourf(
                    X, Y, Z, 
                    levels=np.linspace(layer_min_values[pos], layer_max_values[pos], 100), 
                    cmap='bwr', 
                    vmin=-max_abs_values[pos], 
                    vmax=max_abs_values[pos]
                    )
                ax2.set_xlabel('dimension')
                ax2.set_ylabel('training step')
                ax2.set_title(f'Vanilla Biases (Layer {pos})' if col == 0 else f'{fullBN_position}{ablation_labels.get(fullBN_ablation)} Betas (Layer{pos})')
        # Plotting for beta_position_list
        offset = len(beta_full_list)
        for col, params in enumerate(beta_position_list):
            for row, layer in enumerate(shapes):
                # Create a meshgrid for the neurons and training steps
                neurons = np.arange(layer[0])
                steps = np.arange(0, params[row].shape[0] * 5, 5)
                X, Y = np.meshgrid(neurons, steps)
                Z = params[row]                    
                # sort dimensions by converged value
                if sort_by_converged:
                    Z = Z[:,Z[-1].argsort()]
                ax_idx = offset + col + row * num_cols + 1 if num_rows > 1 else offset + col + 1
                ax2 = fig2.add_subplot(num_rows, num_cols, ax_idx)
                c = ax2.contourf(
                    X, Y, Z, 
                    levels=np.linspace(layer_min_values[layer_position[row]], layer_max_values[layer_position[row]], 100), 
                    cmap='bwr', 
                    vmin=-max_abs_values[layer_position[row]], 
                    vmax=max_abs_values[layer_position[row]]
                    )
                ax2.set_xlabel('dimension')
                ax2.set_ylabel('training step')
                ax2.set_title(f'{position}{ablation_labels.get(ablation_modes[col])} Betas (Layer {layer_position[row]})')
                if col+offset == num_cols - 1:
                     fig2.colorbar(c)

        #fig2.suptitle(f"Comparison of betas across Batch Norm layers:{position} \n (seed {seed}, problem {problem}, lr {lr}), batch size {batch_size}")#, y=0.9)
        plt.savefig(os.path.join(plot_dir, f"{seed}_BetaActivity_Comparison_{position}_contourplot.png"), dpi=350)
        #plt.show()
        #plt.close()


def compare_contour_betas(sort_by_converged, Vanilla_path, fullBN_path, trained_bn_path, fixed_gamma_path):
    """
    Args:
    sort_by_converged (boolean): if True, sort dimensions by converged value
    Vanilla_path (string): path to model without any BatchNorm layers
    fullBN_path (string): path to model with all BatchNorm layers
    trained_bn_path (string): path to model with BatchNorm layers at a specified position and ablation_mode is trained_bn
    fixed_gamma_path (string): path to model with BatchNorm layers at a specified position and ablation mode is fixed_gamma (because we plot betas here)

    Returns:
    None. 2D plot of the beta values for each layer across training time for Vanilla, fullBN, trained_bn and fixed_gamma models
    """

    # Preprocess data for plots
    log_Vanilla = read_log(Vanilla_path)
    shapes_Vanilla = log_Vanilla["SETTINGS"]["shapes"][0]

    path_bias = get_file_path(Vanilla_path, "bias.h5")
    biases = get_parameter_values(path_bias, shapes_Vanilla)
    bias_values_transposed = [bias.T for bias in biases]

    log_fullBN = read_log(fullBN_path)
    shapes_fullBN = log_fullBN["SETTINGS"]["shapes"][0]
    fullBN_position = log_fullBN["PARAMETERS"]["POSITION"][0]
    fullBN_ablation = log_fullBN["PARAMETERS"]["ABLATION"][0]

    path_full_beta = get_file_path(fullBN_path, "beta.h5")
    full_betas = get_parameter_values(path_full_beta, shapes_fullBN)
    full_beta_values_transposed = [beta.T for beta in full_betas]

    log_trained = read_log(trained_bn_path)
    shapes = log_trained["SETTINGS"]["shapes"][0]
    position = log_trained["PARAMETERS"]["POSITION"][0]
    seed = log_trained["PARAMETERS"]["RANDOM_SEED"][0]
    num_epochs = log_trained["SETTINGS"]["num_epochs"][0]
    problem = log_trained["PARAMETERS"]["PROBLEM"][0]
    lr = log_trained["SETTINGS"]["opt_hpars"].apply(lambda x: x["lr"])[0]
    batch_size = log_trained["SETTINGS"]["batch_size"][0]
    ablation_trained = log_trained["PARAMETERS"]["ABLATION"][0]

    position_str = ''.join(str(i) for i in position)
    layer_position = [i+1 for i, val in enumerate(position_str.split(',')) if val == '1']       

    path_trained_beta = get_file_path(trained_bn_path, "beta.h5")
    trained_beta = get_parameter_values(path_trained_beta, shapes)
    trained_beta_values_transposed = [beta.T for beta in trained_beta]  

    log_fixedG = read_log(fixed_gamma_path)
    ablation_fixedG = log_fixedG["PARAMETERS"]["ABLATION"][0]
    path_fixedG_beta = get_file_path(fixed_gamma_path, "beta.h5")
    fixedG_beta = get_parameter_values(path_fixedG_beta, shapes)
    fixedG_beta_values_transposed = [beta.T for beta in fixedG_beta]

    # Create data lists for plotting
    beta_full_list = [bias_values_transposed, full_beta_values_transposed]
    beta_position_list = [trained_beta_values_transposed, fixedG_beta_values_transposed]
    ablation_modes = [ablation_trained, ablation_fixedG]

    # Create output dir
    name = "ParameterActivity_Comparison"
    plot_dir = create_plot_dir(seed, problem, num_epochs, lr, name)

    num_cols = len(beta_full_list) + len(beta_position_list)
    num_rows = max(1,len(shapes))

    ablation_labels = {
        'trained_bn': '$_\\beta, _\\gamma$',
        'fixed_bn': '$_-$',
        'fixed_gamma': '$_\\beta$',
        'fixed_beta': '$_\\gamma$'
        }
    # VScales to use if colormap should be consistent
    # Compute min and max values for each layer across all plots
    layer_min_values = {}
    layer_max_values = {}
    max_abs_values = {}

    # For beta_full_list
    for params in beta_full_list:
        for i, pos in enumerate(layer_position):
            Z = params[pos-1]
            if pos not in layer_min_values:
                layer_min_values[pos] = np.min(Z)
            else:
                layer_min_values[pos] = min(layer_min_values[pos], np.min(Z))
            if pos not in layer_max_values:
                layer_max_values[pos] = np.max(Z)
            else:
                layer_max_values[pos] = max(layer_max_values[pos], np.max(Z))
            if pos not in max_abs_values:
                max_abs_values[pos] = np.max(np.abs(Z))
            else:
                max_abs_values[pos] = max(max_abs_values[pos], np.max(np.abs(Z)))

    # For beta_position_list
    for params in beta_position_list:
        for i, layer in enumerate(shapes):
            Z = params[i]
            pos = layer_position[i]
            layer_min_values[pos] = min(layer_min_values[pos], np.min(Z))
            layer_max_values[pos] = max(layer_max_values[pos], np.max(Z))
            max_abs_values[pos] = max(max_abs_values[pos], np.max(np.abs(Z)))



    # Create 2D contour plots using matplotlib
    with plt.rc_context({**bundles.aistats2022(family="serif"), **tplt.axes.lines()}):
        tplt.axes.lines()
        plt.rcParams.update(tplt.axes.tick_direction(x="inout", y="in"))
        plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
        plt.rcParams.update(tplt.figsizes.neurips2021(nrows=2, ncols=3))
        plt.rcParams.update(tplt.fontsizes.neurips2021())
            
        fig = plt.figure(figsize=(15, 3*max(1,len(shapes))))
        # Plotting for beta_full_list
        for col, params in enumerate(beta_full_list):
            for i, pos in enumerate(layer_position):
                # Create a meshgrid for the neurons and training steps
                neurons = np.arange(shapes[i][0])
                steps = np.arange(0, params[pos-1].shape[0] * 5, 5)#[:100]###############################################################################################################################################################
                X, Y = np.meshgrid(neurons, steps)
                Z = params[pos-1]
                # sort dimensions by converged value
                if sort_by_converged:
                    Z = Z[:,Z[-1].argsort()]
                
                #Z = Z[:100] ###############################################################################################################################################################

                ax_idx = col + i * num_cols + 1 if num_rows > 1 else col + 1
                ax = fig.add_subplot(num_rows, num_cols, ax_idx)
                c = ax.contourf(
                    X, Y, Z, 
                    levels=np.linspace(layer_min_values[pos], layer_max_values[pos], 100), 
                    cmap='bwr', 
                    vmin=-max_abs_values[pos], 
                    vmax=max_abs_values[pos]
                    )
                ax.set_xlabel('dimension')
                ax.set_ylabel('training step')
                ax.set_title(f'Vanilla Biases (Layer {pos})' if col == 0 else f'{fullBN_position}{ablation_labels.get(fullBN_ablation)} Betas (Layer{pos})')
        # Plotting for beta_position_list
        offset = len(beta_full_list)
        for col, params in enumerate(beta_position_list):
            for row, layer in enumerate(shapes):
                # Create a meshgrid for the neurons and training steps
                neurons = np.arange(layer[0])
                steps = np.arange(0, params[row].shape[0] * 5, 5)#[:100]###############################################################################################################################################################
                X, Y = np.meshgrid(neurons, steps)
                Z = params[row]                    

                #Z = Z[:100] ###############################################################################################################################################################

                # sort dimensions by converged value
                if sort_by_converged:
                    Z = Z[:,Z[-1].argsort()]
                ax_idx = offset + col + row * num_cols + 1 if num_rows > 1 else offset + col + 1
                ax = fig.add_subplot(num_rows, num_cols, ax_idx)
                c = ax.contourf(
                    X, Y, Z, 
                    levels=np.linspace(layer_min_values[layer_position[row]], layer_max_values[layer_position[row]], 100), 
                    cmap='bwr', 
                    vmin=-max_abs_values[layer_position[row]], 
                    vmax=max_abs_values[layer_position[row]]
                    )
                ax.set_xlabel('dimension')
                ax.set_ylabel('training step')
                ax.set_title(f'{position}{ablation_labels.get(ablation_modes[col])} Betas (Layer {layer_position[row]})')
                if col+offset == num_cols - 1:
                    fig.colorbar(c)

        #fig.suptitle(f"Comparison of Betas across Batch Norm layers: {position}  \n (seed {seed}, problem {problem}, lr {lr}), batch size {batch_size}")#, y=0.9)
        plt.savefig(os.path.join(plot_dir, f"{seed}_BetaActivity_Comparison_{position}_contourplot.png"), dpi=350)
        #plt.show()
#contour_betas(True, input_paths[0], input_paths[1], input_paths[2], input_paths[3])



