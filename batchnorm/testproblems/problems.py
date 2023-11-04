#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""
import warnings

import torch
import deepobs

# Datasets
from deepobs.pytorch.datasets import cifar10
from deepobs.pytorch.datasets import cifar100

# Net
from .bn_models import Net_cifar10_3c3d_BN
from .bn_models import Net_cifar100_allcnnc_BN
from deepobs.pytorch.testproblems.testproblems_modules import net_cifar10_3c3d

# Testproblems
from deepobs.pytorch.testproblems import cifar10_3c3d
from .bn_tp_cifar10_3c3d import bn_cifar10_3c3d
from .bn_tp_cifar100_3c3d import bn_cifar100_3c3d
from .bn_tp_cifar100_cnnc import bn_cifar100_allcnnc

        
# ##############################################################################
# # CIFAR10/100 3c3d with and without BN
# # ##############################################################################

class cifar10_3c3d_bn(bn_cifar10_3c3d):
    """ 3c3d testproblem from deepobs superclass for cifar10 dataset """
    
    def __init__(self, batch_size, position, l2_reg=0.002):
        """
        """
        super().__init__(batch_size, position, l2_reg)
    
    def set_up(self):
        """
        """
        self.data = cifar10(self._batch_size)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = Net_cifar10_3c3d_BN(num_outputs=10, position=self._position)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

        print(self.net)
    
    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._l2_reg
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if "bias" not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict


class cifar100_3c3d_bn(bn_cifar100_3c3d):
    """ 3c3d testproblem from deepobs superclass for cifar10 dataset """
    
    def __init__(self, batch_size, position, l2_reg=0.002):
        """
        """
        super().__init__(batch_size, position, l2_reg)
    
    def set_up(self):
        """
        """
        self.data = cifar100(self._batch_size)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = Net_cifar10_3c3d_BN(num_outputs=100, position=self._position)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

        print(self.net)
    
    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._l2_reg
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if "bias" not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict


# ##############################################################################
# # CIFAR10/100 All-CNN-C with BN
# ##############################################################################

class cifar100_allcnnc_bn(bn_cifar100_allcnnc):
    """ Like superclass, but uses own net class and batch normalization """
    def __init__(self, batch_size, position, l2_reg=0.0005):
        """
        """
        super().__init__(batch_size, position, l2_reg)
    
    def set_up(self):
        """
        """
        self.data = cifar100(self._batch_size)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = Net_cifar100_allcnnc_BN(position=self._position)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

        print(self.net)

    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._l2_reg
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if "bias" not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict
