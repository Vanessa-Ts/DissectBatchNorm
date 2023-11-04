#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import random
import os
# for omegaconf
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass, field
from itertools import repeat
#
from omegaconf import OmegaConf, MISSING
import torch
import numpy as np
#
from deepobs.config import set_data_dir
#
from configs.config_io import get_config
from color_logging import JsonColorLogger, make_timestamp
from testproblems import get_obs_testproblem, get_obs_BNtestproblem
from incremental_hdf5 import IncrementalHDF5
import optimizers
from testproblems.bn_models import get_model_params, get_model_bias

import matplotlib.pyplot as plt

##############################################################################
# # HELPERS
# ##############################################################################
def create_hdf5(path, size, dtype=np.float32):
    """
    """
    result = IncrementalHDF5(path, size, dtype=np.float32, compression="lzf",
                             data_chunk_length=size, metadata_chunk_length=size,
                             err_if_exists=True)
    return result


# ##############################################################################
# # CLI
#
# input statement: python bn_position_cifar10_3c3d.py --ABLATION="fixed_bn" --POSITION=[1,1,1,0,0]
# 
# ##############################################################################
class AblationOption(Enum):
    trained_bn = "trained_bn"
    fixed_bn = "fixed_bn"
    fixed_beta = "fixed_beta"
    fixed_gamma = "fixed_gamma"



@dataclass
class ConfDef:
    """
    :cvar OBS_DATASETS: Path where the OBS datasets are stored
    :cvar TUNING_CONF: Path to our YAML optimizer config file
    :cvar OPTIMIZER: Which optimizer from TUNING_CONF to use.
      Supported at the moment: SGD
    :cvar PROBLEM: Which DeepOBS problem to train on
    :cvar OUTPUT_DIR: Where to store logs and results
    :cvar MAX_STEPS: If given, train for only this many batches
    :cvar ABLATION: Which of the four different ablation modes: trained_bn, fixed_bn, fixed_beta, fixed_gamma
    :cvar POSITION: At which position should the bn layers be inserted, encode with 0/1 per layer eg. [1,1,1,0,0]
    """
    OBS_DATASETS: str = "batchnorm/output/deepobs_datasets"
    TUNING_CONF: str = "configs/basic_config.yaml"
    OPTIMIZER: str = "SGD"
    PROBLEM: str = "cifar10_3c3d_bn"
    #
    RANDOM_SEED: int = MISSING
    MAX_STEPS: Optional[int] = None
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    #
    OUTPUT_DIR: str = "output"
    LOG_EVERY_STEPS: int = 5

    ABLATION: str = field(default=MISSING, metadata={'validate': lambda x: x in AblationOption._value2member_map_})   
    POSITION: str = MISSING

    

# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == '__main__':
    conf = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    timestamp = make_timestamp(timezone="Europe/Berlin", with_tz_output=False)

    str_position = conf.POSITION.split(",")
    position = [int(i) for i in str_position]

    # retrieve setting from YAML file
    ((batch_size, num_epochs), (opt_name, opt_hpars),
     (sched_name, sched_hpars)) = get_config(
         conf.TUNING_CONF, conf.PROBLEM, conf.OPTIMIZER)
    lr = opt_hpars["lr"]

    # create directory to store results for this run
    run_dir = os.path.join(
        conf.OUTPUT_DIR,
        "__".join([str(conf.RANDOM_SEED), conf.PROBLEM, conf.POSITION + "/Lr_" + str(lr) + "/Ep_"+ str(num_epochs)+"/"+ conf.ABLATION]))

    os.makedirs(run_dir, exist_ok=True)

    txt_logger = JsonColorLogger(f"[{os.path.basename(__file__)}]", run_dir)
    txt_logger.loj("PARAMETERS", OmegaConf.to_container(conf))

    # Avoid re-downloading dataset on local dir
    set_data_dir(conf.OBS_DATASETS)


    # Set up problem and optimizer
    (model, train_loader, loss_fn, losses_fn, eval_fn,
     tproblem) = get_obs_BNtestproblem(conf.PROBLEM, batch_size, position, conf.RANDOM_SEED)


    model.eval()  # for deterministic behavior
    # if conf.INITIAL_SNAPSHOT_PATH is not None:
    #     load_model_params(model, conf.INITIAL_SNAPSHOT_PATH, eval_phase=True)
    model = model.to(conf.DEVICE)


    # set up HDF5 databases to store network parameters (including BN)
    rundir_base = os.path.basename(run_dir)

    steps_per_epoch = len(train_loader)

    # if position contains not only zeros, no matter the size of the position array
    if any(x == 1 for x in position):
        layers, shapes, biases, weights = get_model_params(model)

        beta_params_h5 = create_hdf5(
            os.path.join(run_dir, f"{timestamp}_beta.h5"), len(biases))
        gamma_params_h5 = create_hdf5(
            os.path.join(run_dir, f"{timestamp}_gamma.h5"), len(weights))
        #
        txt_logger.loj("SETTINGS",
                    {"batch_size": batch_size, "num_epochs": num_epochs,
                        "max_steps": conf.MAX_STEPS,
                        "steps_per_epoch": steps_per_epoch,
                        "opt_name": opt_name, "opt_hpars": opt_hpars,
                        "layers": [n for n, _ in layers],
                        "shapes": shapes,
                        "sched_name": sched_name, "sched_hpars": sched_hpars})
    elif all(x == 0 for x in position):
        layers, shapes, biases = get_model_bias(model)
        bias_params_h5 = create_hdf5(
            os.path.join(run_dir, f"{timestamp}_bias.h5"), len(biases))
        #
        txt_logger.loj("SETTINGS",
                    {"batch_size": batch_size, "num_epochs": num_epochs,
                        "max_steps": conf.MAX_STEPS,
                        "steps_per_epoch": steps_per_epoch,
                        "opt_name": opt_name, "opt_hpars": opt_hpars,
                        "layers": [n for n in layers],
                        "shapes": shapes,
                        "sched_name": sched_name, "sched_hpars": sched_hpars})

    max_steps = (conf.MAX_STEPS if conf.MAX_STEPS is not None else
                 (num_epochs * steps_per_epoch))

    opt_class = getattr(optimizers, "Scheduled" + conf.OPTIMIZER)
    opt = opt_class(model.parameters(), **opt_hpars, lr_sched=None)


    # ##########################################################################
    # # MAIN LOOP
    # ##########################################################################
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        if (conf.MAX_STEPS is not None) and global_step > conf.MAX_STEPS:
            break
        # At beg of each epoch, evaluate and log to Cockpit
        train_loss, train_acc, xv_loss, xv_acc, test_loss, test_acc = eval_fn()
        txt_logger.loj(f"EVAL ROUND",
                       {"epoch": epoch, "global_step": global_step,
                        "train_loss": train_loss, "train_acc": train_acc,
                        "xv_loss": xv_loss, "xv_acc": xv_acc,
                        "test_loss": test_loss, "test_acc": test_acc})

        # Training loop
        tproblem.train_init_op()
        for i, (inputs, targets) in enumerate(iter(train_loader), 1):
            if (conf.MAX_STEPS is not None) and global_step > conf.MAX_STEPS:
                break

            # Note that targets.shape=[32], outputs.shape=[32, 100], but the
            # loss_fn applies "mean" reduction, so the 100 outptus are avgd.
            inputs, targets = inputs.to(conf.DEVICE), targets.to(conf.DEVICE)
            opt.zero_grad()
            # fwpass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses = losses_fn(outputs, targets)

            # add individual loss data to logdict and BN params to HDF5 file
            if (global_step % conf.LOG_EVERY_STEPS) == 0:
                current_lr = opt.param_groups[0]["lr"]
                txt_logger.loj("BATCH",
                               {"global_step": global_step,
                                "epoch": epoch,
                                "batch_loss": loss.item(),
                                "lr": current_lr})
                
                if any(x == 1 for x in position):
                    # gather params
                    _, _, biases, weights = get_model_params(model)
                    # append biases and weights to HDF5 file
                    beta_params_h5.append(biases[:, None], str(
                        {"step": global_step,
                        "type": "Model(bias)"}))
                    gamma_params_h5.append(weights[:, None], str(
                        {"step": global_step,
                        "type": "Model(weights)"}))
                elif all(x == 0 for x in position):
                    _, _, biases = get_model_bias(model)
                    bias_params_h5.append(biases[:, None], str(
                        {"step": global_step,
                        "type": "Model(bias)"}))
            #
            loss.backward()
            opt.step()

            # Apply chosen ablation mode to the respective BN params 
            if conf.ABLATION == "fixed_bn" or conf.ABLATION == "fixed_beta":
                for n, m in model.named_modules():
                    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                        m.bias.data[:] = 0
            if conf.ABLATION == "fixed_bn" or conf.ABLATION == "fixed_gamma":
                for n, m in model.named_modules():
                    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                        m.weight.data[:] = 1

            global_step += 1
    # final evaluation
    train_loss, train_acc, xv_loss, xv_acc, test_loss, test_acc = eval_fn()
    txt_logger.loj(f"FINAL EVALUATION",
                   {"epoch": num_epochs, "global_step": global_step,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "xv_loss": xv_loss, "xv_acc": xv_acc,
                    "test_loss": test_loss, "test_acc": test_acc})
    # save final model
    torch.save(model.state_dict(), os.path.join(
        run_dir, f"{timestamp}__{rundir_base}__final_model.pth"))
    
