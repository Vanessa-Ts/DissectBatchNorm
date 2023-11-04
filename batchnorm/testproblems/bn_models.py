#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


from collections import OrderedDict
import torch
import torch.nn as nn 
import numpy as np
from deepobs.pytorch.testproblems.testproblems_utils import tfmaxpool2d, tfconv2d, mean_allcnnc


# ##############################################################################
# # HELPERS
# ##############################################################################

# Get Gamma and Beta params from BN layers
def get_model_params(model):
    """
    """
    biases= [] # depending on the model biases = beta params
    weights = [] # depending on the model weights = gamma params

    layers = [(n, l) for n, l in model.named_modules() if "norm" in n]

    layer_shapes = [tuple(l.bias.shape) for _, l in layers]

    for n, bn in layers:
        bias = torch.cat([
            bn.bias.flatten().detach()]).cpu().numpy()
        weight = torch.cat([
            bn.weight.flatten().detach()]).cpu().numpy() 
        biases.append(bias)
        weights.append(weight)
    biases = np.concatenate(biases)
    weights = np.concatenate(weights)

    return  layers, layer_shapes, biases, weights

# Get biases from Vanilla model (w/o BN layers)
def get_model_bias(model):
    """
    """
    conv_biases= [] # depending on the layer biases
    dense_biases = [] 
    conv_layers = []
    conv_layer_shapes = []
    
    dense_layers = [(n, l) for n, l in model.named_modules() if "dense" in n]
    dense_layer_shapes = [tuple(l.bias.shape) for _, l in dense_layers]

    # extract conv layers 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
            conv_bias = torch.cat([
                module.bias.flatten().detach()]).cpu().numpy()
            conv_biases.append(conv_bias)
            conv_layer_shapes.append(tuple(module.bias.shape))

    for n, m in dense_layers:
        dense_bias = torch.cat([
            m.bias.flatten().detach()]).cpu().numpy()
        dense_biases.append(dense_bias)

    layers = conv_layers + [n for n, _ in dense_layers]
    layer_shapes = conv_layer_shapes + dense_layer_shapes
    biases = conv_biases + dense_biases
    biases = np.concatenate(biases)

    return  layers, layer_shapes, biases


##############################################################################################################
# 3C3D Model
##############################################################################################################  


class Net_cifar10_3c3d_BN(torch.nn.Sequential):
    """
        Basic 3c3d network with batch normalization applied before ReLU activation.
        Position argument denotes position and number of BN layers in the network.
    """

    def __init__(self, num_outputs=10, position=[0,0,0,0,0]):
        
        super(Net_cifar10_3c3d_BN, self).__init__()

        # counter for the position of the BN layers
        bn_counter = 0
        print(position)

        self.add_module(
            "conv1", 
            tfconv2d(
                in_channels=3, 
                out_channels=64, 
                kernel_size=5,
                )

        )
        if position[bn_counter] == 1:
            self.add_module("norm1", nn.BatchNorm2d(64))
            self.conv1.bias = None
        bn_counter += 1
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "maxpool1",
            tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same"),
        )

        self.add_module(
            "conv2", 
            tfconv2d(
                in_channels=64, 
                out_channels=96, 
                kernel_size=3,
                )
        )
        if position[bn_counter] == 1:
            self.add_module("norm2", nn.BatchNorm2d(96))
            self.conv2.bias = None
        bn_counter += 1
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "maxpool2",
            tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same"),
        )

        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm3", nn.BatchNorm2d(128))
            self.conv3.bias = None
        bn_counter += 1
        self.add_module("relu3", nn.ReLU())
        self.add_module(
            "maxpool3",
            tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same"),
        )

        self.add_module("flatten", torch.nn.Flatten())
        self.add_module(
            "dense1", nn.Linear(in_features=3 * 3 * 128, out_features=512)
        )
        if position[bn_counter] == 1:
            self.add_module("norm4", nn.BatchNorm1d(512))
            self.dense1.bias = None
        bn_counter += 1
        self.add_module("relu4", nn.ReLU())
        self.add_module("dense2", nn.Linear(in_features=512, out_features=256))
        if position[bn_counter] == 1:
            self.add_module("norm5", nn.BatchNorm1d(256))
            self.dense2.bias = None
        self.add_module("relu5", nn.ReLU())
        self.add_module(
            "dense3", nn.Linear(in_features=256, out_features=num_outputs)
        )



        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.00)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)  # gamma
                nn.init.zeros_(module.bias)  # beta
                nn.init.zeros_(module.running_mean) # mean
                nn.init.constant_(module.running_var, 1.0) # var
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.00) 


##############################################################################################################
# AllCNNC 
##############################################################################################################  

class Net_cifar100_allcnnc_BN(nn.Sequential):   
    """
        All-CNN-C network with batch normalization applied before ReLU activation
        Position argument denotes position and number of BN layers in the network.
    """

    def __init__(self, num_outputs=100, position=[0,0,0,0,0,0,0,0]):
        super(Net_cifar100_allcnnc_BN, self).__init__()

        # counter for the position of the BN layers
        bn_counter = 0

        self.add_module("dropout1", nn.Dropout(p=0.2))

        self.add_module(
            "conv1",
            tfconv2d(
                in_channels=3, 
                out_channels=96, 
                kernel_size=3, 
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm1", nn.BatchNorm2d(96))
            self.conv1.bias = None
        bn_counter += 1
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=96, 
                out_channels=96, 
                kernel_size=3, 
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm2", nn.BatchNorm2d(96))
            self.conv2.bias = None
        bn_counter += 1
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm3", nn.BatchNorm2d(96))
            self.conv3.bias = None
        bn_counter += 1
        self.add_module("relu3", nn.ReLU())

        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module(
            "conv4",
            tfconv2d(
                in_channels=96, 
                out_channels=192, 
                kernel_size=3, 
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm4", nn.BatchNorm2d(192))
            self.conv4.bias = None
        bn_counter += 1
        self.add_module("relu4", nn.ReLU())
        self.add_module(
            "conv5",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm5", nn.BatchNorm2d(192))
            self.conv5.bias = None
        bn_counter += 1
        self.add_module("relu5", nn.ReLU())
        self.add_module(
            "conv6",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm6", nn.BatchNorm2d(192))
            self.conv6.bias = None
        bn_counter += 1
        self.add_module("relu6", nn.ReLU())

        self.add_module("dropout3", nn.Dropout(p=0.5))

        self.add_module(
            "conv7", tfconv2d(
                in_channels=192, 
                out_channels=192, 
                kernel_size=3)
        )
        if position[bn_counter] == 1:
            self.add_module("norm7", nn.BatchNorm2d(192))
            self.conv7.bias = None
        bn_counter += 1
        self.add_module("relu7", nn.ReLU())
        self.add_module(
            "conv8",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=1,
                tf_padding_type="same",
            ),
        )
        if position[bn_counter] == 1:
            self.add_module("norm8", nn.BatchNorm2d(192))
            self.conv8.bias = None
        bn_counter += 1
        self.add_module("relu8", nn.ReLU())
        self.add_module(
            "conv9",
            tfconv2d(
                in_channels=192,
                out_channels=num_outputs,
                kernel_size=1,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu9", nn.ReLU())

        self.add_module("mean", mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0) # changed from 0.1 to 0.0 for comparison with beta
                nn.init.xavier_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)  # gamma
                nn.init.zeros_(module.bias)  # beta
                nn.init.zeros_(module.running_mean) # mean
                nn.init.constant_(module.running_var, 1.0) # var