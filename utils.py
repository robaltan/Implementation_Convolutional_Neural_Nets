import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        """
        An implementation of a max-pooling layer.

        Parameters:
        - kernel_size: the size of the window to take a max over
        """
        self.kernel_size = kernel_size 

        # Given there is no stride size given, let's assume that stride 
        # is the same as one of the dimensions of the kernel
        self.stride = self.kernel_size

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """
        # Retrieve the batch size, number of input channels, height and width of each channel
        batch_size, number_of_input_channels, height, width = x.size()[0], x.size()[1], x.size()[2], x.size()[3]

        # Retrieve kernel height and width
        kernel_height, kernel_width = self.kernel_size, self.kernel_size

        # Perform assertion that height is divisible by kernel height and same
        # for width
        assert height % kernel_height == 0
        assert width % kernel_width == 0
        
        # Reshape x so that max could be found by two dimensions
        x = x.reshape((batch_size, number_of_input_channels, 
                      height // kernel_height, kernel_height, width // kernel_width, kernel_width)) 
        # Perform max operation on two dimensions 
        out = torch.amax(x, dim=(3, 5))
        return out

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        # Assign the number of input and output channels to variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Assign user defined by bias enabled to a variable
        self.bias_enabled = bias

        # Initialize an empty tensor with the size (C, F)
        self.weights = torch.empty((out_channels, in_channels))

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

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """
        # Make sure that the height of the input channel is equivalent to
        # the number of input channels 
        assert x.size()[-1] == self.in_channels
        
        # Apply the forward pass
        out = torch.matmul(x, self.weights.T)
        if self.bias_enabled:
          # Add bias if user enabled bias
          out = out + self.bias
        

        # Make sure that the height of the output channel is equivalent to
        # the number of output channels
        assert out.size()[-1] == self.out_channels

        return out