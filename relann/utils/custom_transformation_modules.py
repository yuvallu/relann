import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

###############################################################################
# Custom Transformation Modules
###############################################################################


class OnlySecondArgLinear(nn.Module):
    """
    A transformation module that takes two tensor arguments.
    It applies a linear transformation only on the second tensor.
    """

    def __init__(self, input_dim, output_dim):
        super(OnlySecondArgLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, tensor1, tensor2):
        # tensor1 is passed through unchanged.
        transformed_tensor2 = self.linear(tensor2)
        return transformed_tensor2


class SumReLU(nn.Module):
    """
    This module expects multiple input vectors (tensors) as arguments.
    It sums them elementwise and applies the ReLU activation function.
    """

    def __init__(self):
        super(SumReLU, self).__init__()

    def forward(self, *inputs):
        # Sum all provided tensors elementwise.
        summed = sum(inputs)
        # Apply ReLU activation to the summed tensor.
        return torch.relu(summed)


class Concat2AndLinear(nn.Module):
    def __init__(self, input_dim=64):
        super(Concat2AndLinear, self).__init__()
        # Define a linear layer that transforms a 64-dimensional input into a 1-dimensional output.
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x1, x2):
        # Assume x1 and x2 are tensors of shape (batch_size, 32).
        # Concatenate the two tensors along the feature dimension (dim=1) to form a tensor of shape (batch_size, 64).
        concatenated = torch.cat((x1, x2), dim=1)
        # Apply the linear layer to the concatenated tensor.
        output = self.linear(concatenated)
        return output


# Re-export from tensor_term_compiler so existing imports from this module still work
from relann.tensor_term_compiler import ArgMax
