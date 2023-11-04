#!/bin/bash


# ##############################################################################
# # Slurm submission file, allows to run large arrays of jobs defined via
# # nested loops (see bottom)
# #
# # Note that SLURM will stop reading SBATCH directives as soon as there
# # is executable code (this includes defining variables and any non-comment),
# # so make sure to leave that for the end.
# # More info: https://slurm.schedmd.com/sbatch.html
# ##############################################################################

# # SET JOB TITLE HERE
#SBATCH --job-name="OneJob_Cifar100_ALL"


# ##############################################################################
# # LOGGING
# # Add mail address to be notified 
# ##############################################################################

# # Notify via email
# # https://slurm.schedmd.com/sbatch.html#OPT_mail-type
# # NONE,BEGIN,END,FAIL,ALL,REQUEUE,TIME_LIMIT,TIME_LIMIT_80, ARRAY_TASKS
#SBATCH --mail-type=END,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=SET MAIL ADRESS HERE

# # output pathing for logs. The directory must exist, otherwise no logs
# # More info: https://slurm.schedmd.com/sbatch.html#lbAH
# # Change the following paths for your folder structure:


#SBATCH --error /mnt/qb/work/hennig/hmx169/logs_slurm/%u_%x_%10A_%4a.logerr
#SBATCH --output /mnt/qb/work/hennig/hmx169/logs_slurm/%u_%x_%10A_%4a.log


# ##############################################################################
# # RESOURCES
# # Overview: https://stackoverflow.com/a/54845007
# # The following info can be useful to decide what to request
# # list partitions: `sinfo -s`
# # partition details: `scontrol show partitions`
# # partition info example: `sinfo --long --partition=gpu-2080ti`
# # further inspection: `scontrol show partition gpu-2080ti-interactive`
# # States are:
# #   * alloc (allocated to one or more jobs)
# #   * mix (some of its CPUs allocated, others are IDLE)
# #   * idle (no jobs allocated)
# #   * down/drain/draining (no touchy)
# # check `man sinfo` for comprehensive info on nodes
# #
# # Also check defaults for a given partition, e.g.:
# # general max defaults: scontrol show config | grep -i "max"
# # partition defaults: scontrol show partition gpu-2080ti
# #
# # Billing weights: scontrol show config | grep -i "priorityweight"
# ##############################################################################

# fixed
# ntasks is hardcoded to 1. Use arrays for parallelism
# 1 node means all cores on same machine
#SBATCH --nodes=1
#SBATCH --ntasks=1

# change these
#SBATCH --partition=gpu-2080ti # -preemptable  # sinfo -s
#SBATCH --time=1-18:00  # D-HH:MM  MODIFY THIS IF JOB NEEDS MORE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


# ##############################################################################
# # MAIN ROUTINE
# # note the following array syntax:
# # --array=0-9  --> 10 jobs, from 0 to 9 (both included)
# # --array=4,7,123  --> 3 jobs with numbers 4, 7 and 123
# # --array=0-4:2  --> 3 jobs with indices 0,2 and 4
# #
# # If you call sbatch via "arg1=val1 arg2=val2 sbatch test.sh", you can then
# # access those variables here via "./myprog $arg1 $arg2"
# ##############################################################################

# Nested array loop: we need to specify first total number of iterations!
# both ends of the range are included, so go from 1 to (n1*n2*n3...)
# credit: https://stackoverflow.com/a/41934241


# Now define grid search domain: multiple runs per problem




TUNING_CONF="configs/basic_config.yaml"
OPTIMIZER="SGD"
OUTPUT_DIR="output"
PROBLEM="cifar100_allcnnc_bn"
ABLATION="trained_bn"
POSITION="0,0,0,0,0,0,0,0"


python -u run_trainingLoop.py \
        TUNING_CONF=$TUNING_CONF \
        OPTIMIZER=$OPTIMIZER \
        OUTPUT_DIR=$OUTPUT_DIR \
        PROBLEM=$PROBLEM \
        ABLATION=$ABLATION \
        POSITION=$POSITION

#done




