import math
from functools import partial
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.act_fn = ReLU_full_grad()

        if not self.opt.model.convolutional:
            self.num_channels = [getattr(self.opt.model.fully_connected, f"hidden_dim_{i+1}", 2000)
                                 for i in range(self.opt.model.num_blocks)]

            # Initialize the model.
            self.model = nn.ModuleList()
            prev_dimension = opt.input.input_size
            for i in range(len(self.num_channels)):
                block = nn.ModuleList([nn.Linear(prev_dimension, self.num_channels[i])])
                for j in range(self.opt.model.fully_connected.num_layers_per_block - 1):
                    block.append(nn.Linear(self.num_channels[i], self.num_channels[i]))
                prev_dimension = self.num_channels[i]
                self.model.append(block)
        else:
            self.num_channels = [getattr(self.opt.model.conv, f"channels_{i+1}", 1) * (getattr(self.opt.model.conv, f"output_size_{i+1}", 1)**2)
                                 for i in range(self.opt.model.num_blocks)]

            self.model = nn.ModuleList()
            prev_dimension = self.opt.model.conv.input_channels
            for i in range(self.opt.model.num_blocks):
                # self.model.append(nn.ModuleList([LocallyConnected2d(prev_dimension,  # in_channels
                #                                                     getattr(self.opt.model.conv, f"channels_{i+1}", 1),     # out_channels
                #                                                     getattr(self.opt.model.conv, f"output_size_{i+1}", 1),  # output_size
                #                                                     getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                #                                                     getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                #                                                     getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                #                                                     )])) 
                self.model.append(nn.ModuleList([nn.Conv2d(prev_dimension,                                       # in_channels
                                                        getattr(self.opt.model.conv, f"channels_{i+1}", 1),     # out_channels
                                                        getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                                                        stride=getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                                                        padding=getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                                                        )]))
                prev_dimension = getattr(self.opt.model.conv, f"channels_{i+1}", 1)

        for block in self.model:
            for layer in block:
                # print(layer.weight.shape[0] * layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3] * layer.weight.shape[4] * layer.weight.shape[5])
                print(layer.weight.shape)

        # Initialize downstream classification loss.
        if self.opt.model.convolutional and self.opt.model.num_blocks in self.opt.model.conv.pool:
            self.linear_classifier = nn.Sequential(
                nn.Linear(self.num_channels[-1]//4, opt.input.num_classes, bias=False)
            )
        else:
            self.linear_classifier = nn.Sequential(
                nn.Linear(self.num_channels[-1], opt.input.num_classes, bias=False)
            )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for block in self.model:
            for m in block:
                if not self.opt.model.convolutional:
                    if isinstance(m, nn.Linear):
                        if self.opt.training.init == "He":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[1])
                            )
                        elif self.opt.training.init == "Xavier":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[1])
                            )
                        torch.nn.init.zeros_(m.bias)
                else:
                    if isinstance(m, LocallyConnected2d):
                        if self.opt.training.init == "He":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[-1]*m.weight.shape[-4])
                                # m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[-2]*m.weight.shape[-3]*m.weight.shape[-5])
                            )
                        elif self.opt.training.init == "Xavier":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[-1]*m.weight.shape[-4])
                                # m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[-2]*m.weight.shape[-3]*m.weight.shape[-5])
                            )
                        torch.nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Conv2d):
                        if self.opt.training.init == "He":
                            # torch.nn.init.normal_(
                            #     m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                            # )
                            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        elif self.opt.training.init == "Xavier":
                            # torch.nn.init.normal_(
                            #     m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                            # )
                            torch.nn.init.xavier_normal_(m.weight)
                        torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=list(range(1,len(z.shape))), keepdim=True)) + eps)
    
    def forward(self, inputs, labels):
        scalar_outputs = {}

        # Concatenate positive and negative samples and create corresponding labels.
        x = inputs['sample']
        if not self.opt.model.convolutional:
            x = x.reshape(x.shape[0], -1)
        x = self._layer_norm(x)

        for block_idx, block in enumerate(self.model):

        #     # backward for two layers

            x = block[0](x)
            x = self.act_fn.apply(x)
            if self.opt.training.dropout > 0:
                x = F.dropout(x, p=self.opt.training.dropout, training=True)
            # x = self._layer_norm(x)
            # x = block[1](x)
            # x = self.act_fn.apply(x)

            x = self._layer_norm(x)

            if self.opt.model.convolutional and (block_idx+1) in self.opt.model.conv.pool:
                x = F.max_pool2d(x, 2, 2)  # maxpool

            print(x.shape)

        output = self.linear_classifier(x.reshape(x.shape[0], -1))
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )
        scalar_outputs["Loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        self.bias = nn.Parameter(
            torch.randn(1, out_channels, output_size[0], output_size[1])
        )
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = (padding, padding, padding, padding)
        
    def forward(self, x):
        _, c, h, w = x.size()
        x = F.pad(x, self.padding, mode='constant', value=0)
        kh, kw = self.kernel_size
        dh, dw = self.stride

        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1]) + self.bias
        return out

class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(input):
        return input.clamp(min=0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.input = inputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        assert input.shape == grad_output.shape
        grad_out = grad_output.clone()
        grad_out[input<0] = 0
        return grad_out
    
    @staticmethod
    def jvp(ctx, grad_input):
        input = ctx.input
        assert input.shape == grad_input.shape
        grad_in = grad_input.clone()
        grad_in[input<0] = 0
        return grad_in