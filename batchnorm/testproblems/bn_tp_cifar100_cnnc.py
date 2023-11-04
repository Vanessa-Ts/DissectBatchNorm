# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for CIFAR-100."""

from torch import nn
from deepobs.pytorch.datasets.cifar100 import cifar100
from .bn_models import Net_cifar100_allcnnc_BN
from .bn_testproblem import BN_WeightRegularizedTestproblem

class bn_cifar100_allcnnc(BN_WeightRegularizedTestproblem):
    """Like DeepOBS test problem class for the All Convolutional Neural Network C
  on Cifar-100, But  with position argument for the BN layers

  Details about the architecture can be found in the `original paper`_.

  The paper does not comment on initialization; here we use Xavier for conv
  filters and constant 0.1 for biases.

  L2-Regularization is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  .. _original paper: https://arxiv.org/abs/1412.6806

  The reference training parameters from the paper are ``batch size = 256``,
  ``num_epochs = 350`` using the Momentum optimizer with :math:`\\mu = 0.9` and
  an initial learning rate of :math:`\\alpha = 0.05` and decrease by a factor of
  ``10`` after ``200``, ``250`` and ``300`` epochs.

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.
  """

    def __init__(self, batch_size, position, l2_reg=0.0005):

        """Create a new All CNN C test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """

        super(bn_cifar100_allcnnc, self).__init__(batch_size, position, l2_reg)

    def set_up(self):
        """Set up the All CNN C test problem on Cifar-100."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = Net_cifar100_allcnnc_BN(position=self._position)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()