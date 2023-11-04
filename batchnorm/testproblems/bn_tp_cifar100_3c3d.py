# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for CIFAR-10."""

from torch import nn
from deepobs.pytorch.datasets.cifar100 import cifar100
from .bn_models import Net_cifar10_3c3d_BN
from .bn_testproblem import BN_WeightRegularizedTestproblem

class bn_cifar100_3c3d(BN_WeightRegularizedTestproblem):
    """
    Like DeepOBS test problem class for for a three convolutional and three dense \
    layered neural network on Cifar-100.
    But with position argument for the BN layers

    - three conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with ``512`` and ``256`` units and ReLU activation
    - 100-unit output layer with softmax
    - cross-entropy loss
    - L2 regularization on the weights (but not the biases) with a default
      factor of 0.002

    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to ``0.0``.

    A working training setting is ``batch size = 128``, ``num_epochs = 100`` and
    SGD with learning rate of ``0.01``.

    Args:
        batch_size (int): Batch size to use.
        position (array): Array of length 5 with the position of the BN layers in the network.
        l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
            is used on the weights but not the biases. Defaults to ``0.002``.

    Attributes:
        data: The DeepOBS data set class for Cifar-100.
        loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
        net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_cifar10_3c3d).

    Methods:
        get_regularization_loss: Returns the current regularization loss of the network state.
    """

    def __init__(self, batch_size, position, l2_reg=0.002):
        """Create a new 3c3d test problem instance on Cifar-10.

        Args:
            batch_size (int): Batch size to use.
            l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
                is used on the weights but not the biases. Defaults to ``0.002``.
        """

        super().__init__(batch_size, position, l2_reg)

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = Net_cifar10_3c3d_BN(num_outputs=100, position=self._position)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    #def get_regularization_groups(self):
    #    """Creates regularization groups for the parameters.

    #    Returns:
    #        dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
    #    """
    #    no, l2 = 0.0, self._l2_reg
    #    group_dict = {no: [], l2: []}

    #    for parameters_name, parameters in self.net.named_parameters():
    #        # penalize only the non bias layer parameters
    #        if "bias" not in parameters_name:
    #            group_dict[l2].append(parameters)
    #        else:
    #            group_dict[no].append(parameters)
    #    return group_dict

    