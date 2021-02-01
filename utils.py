import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
import math

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):

        super(Conv2d, self).__init__()
        """
        An implementation of a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Parameters:
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - kernel_size: Size of the convolving kernel
        - stride: The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - padding: The number of pixels that will be used to zero-pad the input.
        """
  
        # Assign parameters to be used inside the forward pass
        self.kernel_size = kernel_size # kernel size defined by the user
        self.out_channels = out_channels # the number of output channels defined by the user
        self.in_channels = in_channels # the number of input channels defined by the user
        self.padding = padding # the number of pixels defined by the user for zero-padding
        self.stride = stride # the number of pixels defined by the user between adjacent receptive fields 
        self.bias_enabled = bias # a boolean parameter defined by the user for whether bias will be used

        # Initialize an empty tensor with the size (F, C, HH, WW)
        self.weights = torch.empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])) 


        # Use Kaiman Initialization for the weights and the bias
        # Followed how PyTorch implemented it
        # See https://github.com/pytorch/pytorch/blob/316f0b89c3aa51329f40a13c00de587da60faa66/torch/nn/init.py#L67
        fan = self.weights.size(1) * self.weights[0][0].numel()
        gain = math.sqrt(2.0) # gain for the 'relu' function
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
          self.weights.uniform_(-bound, bound)

        if self.bias_enabled:
          # Initialize an empty tensor with the size (F, )
          self.bias = torch.empty((out_channels, ))
          bound = 1 / math.sqrt(fan) # no gain needed
          self.bias.uniform_(-bound, bound)

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """

        # Used the torch documentation to apply the forward pass
        batch_size, in_channels, height, width = x.size()[0], x.size()[1], x.size()[2], x.size()[3] 

        assert self.in_channels == in_channels

        unfolded_x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding,
                     stride=self.stride)
        w_reshaped = self.weights.view(self.out_channels, -1)
        out = w_reshaped @ unfolded_x

        # Using the formula to calculate the output dimensions
        output_height = ((height - self.kernel_size[0] + self.padding * 2) // self.stride) + 1
        output_width = ((width - self.kernel_size[1] + self.padding * 2) // self.stride) + 1

        assert unfolded_x.shape == (batch_size, 1, in_channels * self.kernel_size[0] * self.kernel_size[1], output_height * output_width)


        if self.bias_enabled:
          out = out.view(batch_size, self.out_channels, output_height, output_width) + self.bias.unsqueeze(1).unsqueeze(1)
        else:
          out = out.view(batch_size, self.out_channels, output_height, output_width)

        return out
