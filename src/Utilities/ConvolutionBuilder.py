import copy
import math
import operator
from functools import reduce


def compute_convolutions(num_convolutions, voxel_shape, d_model, convolution_scaling):
    shape = list(copy.deepcopy(voxel_shape))

    dim_channel = shape[-1]
    dim_max = max(shape)
    dim_reduction_per_step = (dim_max - 1) / num_convolutions

    dim_reduction_adjusted = math.floor(dim_reduction_per_step)
    dim_reduction_error = dim_reduction_per_step - dim_reduction_adjusted

    convolutions = []
    error = dim_reduction_error
    for i in range(num_convolutions):

        local_dim_reduction = dim_reduction_adjusted

        if error >= 1:
            error -= 1
            local_dim_reduction += 1
        if i == num_convolutions - 1:
            if error > 0:
                local_dim_reduction += 1

        prev_dim = reduce(operator.mul, shape, 1)

        adj = [local_dim_reduction + 1] * 3
        for i in range(3):
            if (shape[i] <= adj[i]):
                adj[i] = shape[i]
            shape[i] += 1 - adj[i]

        shape[-1] = dim_channel
        post_dim = reduce(operator.mul, shape, 1)
        dims_scale = prev_dim / post_dim
        dim_channel = max(int(dim_channel * dims_scale), 1)
        shape[-1] = dim_channel
        convolutions.append(
            [dim_channel, tuple(adj), 1, None]
        )
        error += dim_reduction_error

    scale_model = d_model / dim_channel
    for i in range(num_convolutions):
        adj_channels = max(int(convolutions[i][0] * scale_model), 1) * convolution_scaling
        convolutions[i][0] = adj_channels
        shape[-1] = adj_channels

    return [[(voxel_shape[-1], 1, 1, None)] + convolutions, shape]
