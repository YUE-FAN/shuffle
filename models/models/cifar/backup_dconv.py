# This code come from https://github.com/oeway/pytorch-deform-conv
# Thank him, and I change some code.

from torch.autograd import Variable
import torch
from torch import nn
import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates

cuda_number = 0


class Dconv_cos(nn.Module):
    """
    re-arrange the spatial layout of the activations.
    Then perform a dilated convolution to achieve the dconv.
    We don't have the concept of the OFFSET at all. Instead, we generate
    the new sampling locations directly. Even though I believe offset is
    easier to obtain in terms of the learning, but our deformation has
    nothing to do with the learning, it is a parameter-free method.

    The key problems in our method are:
    1. How to obtain the new sampling locations.
        For now we choose 9 locations where the activations are most similar
        to the center one. And we discard the spatial distance between them.
        TODO: consider the spatial distance and use other similarity measure
    2. How to arrange a new squared layout for the upcoming convolution.
        For now we just go through all the activations from the left to the
        right and then top down.
        TODO: come up with a more descent arrangement
    """
    def __init__(self, height, width, inplane, outplane, win_range=5, kernel_size=3):
        """

        :param win_range: defines the size of the search window (for now it has to be odd)
        :param height: the height of the input feature map
        :param width: the width of the input feature map
        """
        super(Dconv_cos, self).__init__()
        self.h = height
        self.w = width
        self.win_range = win_range

        # offset dict is a dict of the same length of the number of activations in the input feature map.
        # Each key is a tuple denoting the location of the activation, the value is a list of the
        # indices of its neighbors, the length of the list is win_range * win_range.
        self.offset_dict = {}

        for key_i in range(self.h):
            for key_j in range(self.w):

                idx_list = []  # the list for the indices of the neighbors of the point (key_i, key_j)

                # go through all the positions nearby and add them to the list
                for i in range(self.h):
                    for j in range(self.w):
                        if np.abs(i-key_i) <= (self.win_range-1)/2 and np.abs(j-key_j) <= (self.win_range-1)/2:
                            idx_list.append(self.loc_1D([i, j]))

                self.offset_dict[(key_i, key_j)] = torch.tensor(idx_list).cuda(cuda_number)

        # since each location has its own offset, the re-arranged feature map should have a larger size than the x,
        # namely, x is 32x32 and kernel is 3x3, then x_offset should be 96x96. But only 32x32 locations out of 96x96
        # are needed for convolution. Thus a dilated convolution should be used. The stride should be the same as the
        # size of the kernel and the padding should be disabled. For now, only 3x3 convolution is supported.
        # TODO: support convolutions with any kernel size
        self.kernel_size = kernel_size
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0, bias=False)

        # the relative locations of an activation, those locations has to be filled, it is the same size of the kernel
        self.directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]

    def forward(self, x):
        """
        First compute distance matrix, and then re-arrange x according to it.
        """
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]

        # distance matrix contains the cosine similarity between each 2 locations. If x is of the size [128,3,32,32],
        # the distance matrix will be of size [128, 1024, 1024], 1024 enumerates all possible locations from (0,0) to
        # (31,31) with a row first principal.
        # distance_mat = torch.matmul(x.permute(0, 2, 1), x)  # TODO: only half of the computation here is needed
        # norm_x = torch.diag(distance_mat, diagonal=0)
        # distance_mat = distance_mat / torch.matmul(norm_x.t(), norm_x)

        # assign the first n most similar activations from the neighbors to the squared neighbourhood of "key"
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2] * self.kernel_size, x_shape[3] * self.kernel_size).cuda(cuda_number)

        for batch in range(x_shape[0]):
            for key, idx_list in self.offset_dict.items():
                tmpX1 = x[batch, :, self.loc_1D(key)].view(x_shape[1], 1)  # [3, 1]
                tmpX2 = x[batch, :, idx_list].view(x_shape[1], len(idx_list))  # [3, 7*7]

                distance_mat = torch.matmul(tmpX2.permute(1, 0), tmpX1)  # [7*7, 1]
                norm_x = torch.norm(tmpX1) * torch.norm(tmpX2, dim=0)
                dist_list = distance_mat / norm_x.unsqueeze(-1)  # a vector, same shape with idx_list

                # sort in an ascending way because it is a distant measure
                delll, orders = torch.sort(dist_list.squeeze(-1), descending=False)
                # sort idx_list according to dist_list
                sorted_idx_list = idx_list[orders]
                # take the first several smallest ones
                sorted_idx_list = sorted_idx_list[0: self.kernel_size * self.kernel_size]
                # now sort the idx_list ascendingly
                sorted_idx_list, _ = torch.sort(sorted_idx_list, descending=False)

                for i, relativ_loc in enumerate(self.directions):
                    x_offset[batch, :, (key[0]*self.kernel_size+1)+relativ_loc[0],
                             (key[1]*self.kernel_size+1)+relativ_loc[1]] = x[batch, :, sorted_idx_list[i]]

        # apply dilated convolution so that skip the undefined locations
        x_offset = self.dilated_conv(x_offset)

        return x_offset

    def loc_2D(self, loc_1D):
        # turn a 1D coordinate into a 2D coordinate
        return [loc_1D // self.w, loc_1D % self.w]

    def loc_1D(self, loc_2D):
        # turn a 2D coordinate into a 1D coordinate
        return loc_2D[0] * self.w + loc_2D[1]


class ConvOffset2D_share(nn.Conv2d):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D_share, self).__init__(self.filters, self.filters, 3, padding=1, bias=False, **kwargs)  # TODO:
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()  # [128, 3, 32, 32]

        # y = x.clone()
        # y = y.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]
        # cos = torch.matmul(y.permute(0, 2, 1), y)  # [128, 1024, 1024]



        # (b, c, h, w)
        offset = super(ConvOffset2D_share, self).forward(x)  # [128, 3, 32, 32]
        # a = offsets.contiguous()[0,:,:,:]
        # a = a.view(3, 32*32, 2)
        # print()
        # print(a)
        # (b, c*2, h, w)
        offset = offset.view(x_shape[0]*x_shape[1], x_shape[2]*x_shape[3])  # [128*3, 32*32]
        offset = offset.unsqueeze(-1)  # [128*3, 32*32, 1]
        offsets = torch.cat((offset.contiguous(), offset.contiguous()), 2)  # [128*3, 32, 32, 2]
        offsets = offsets.view(x_shape[0], 2*x_shape[1], x_shape[2], x_shape[3])
        # (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h*w)    grid:(b*c, h*w, 2)
        # grid contains b*c same 2D arrays, which is
        #  [[0 0]
        #   [0 1]
        #   [0 2]
        #   [1 0]
        #   [1 1]
        #   [1 2]]
        # for h=2, w=3
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))
        # x_offset: (b, c, h, w)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_size = x.size(0), (x.size(1), x.size(2))
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_size, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_size, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_size, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters * 2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()  # [128, 3, 32, 32]


        # (b, c*2, h, w)
        offsets = super(ConvOffset2D, self).forward(x)  # [128, 6, 32, 32]


        # (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h*w)    grid:(b*c, h*w, 2)
        # grid contains b*c same 2D arrays, which is
        #  [[0 0]
        #   [0 1]
        #   [0 2]
        #   [1 0]
        #   [1 1]
        #   [1 2]]
        # for h=2, w=3
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))
        # x_offset: (b, c, h, w)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_size = x.size(0), (x.size(1), x.size(2))
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_size, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_size, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_size, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_gather_2d(input, coords):
    inds = coords[:, 0] * input.size(1) + coords[:, 1]
    x = torch.index_select(th_flatten(input), 0, inds)
    return x.view(coords.size(0))


def th_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1
    input_size = input.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_2d(input, coords_lt.detach())
    vals_rb = th_gather_2d(input, coords_rb.detach())
    vals_lb = th_gather_2d(input, coords_lb.detach())
    vals_rt = th_gather_2d(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (bc, s, s)
    coords : tf.Tensor. shape = (bc, s*s, 2)
    Returns
    -------
    tf.Tensor. shape = (bc, s, s)
    """

    batch_size = input.size(0)
    input_size = input.size(1)
    n_coords = coords.size(1)  # n_coords is the h*w
    # print('sadsadsaddsadsadsadsadsadsadsa')
    # print(coords)
    # print('sadsadsaddsadsadsadsadsadsadsa')
    coords = torch.clamp(coords, 0, input_size - 1)  # the range of the offset can cover the whole image
    # turn the fractional locations into the 4 nearest integer locations
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    # print('sadsadsaddsadsadsadsadsadsadsa')
    # print(idx.size())
    # print('sadsadsaddsadsadsadsadsadsadsa')
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda(cuda_number)

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_size, dtype, cuda):
    grid = np.meshgrid(
        range(input_size[0]), range(input_size[1]), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda(cuda_number)
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (bc, s, s)
    offsets: torch.Tensor. shape = (bc, s, s, 2)
    grid: (b*c, h*w, 2), which is the x-y location
    Returns
    -------
    torch.Tensor. shape = (bc, s, s)
    """
    batch_size = input.size(0)
    input_size = [input.size(1), input.size(2)]

    # (bc, h*w, 2)
    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        # grid:(b*c, h*w, 2)
        grid = th_generate_grid(batch_size, input_size, offsets.data.type(), offsets.data.is_cuda)
    # (b*c, h*w, 2)
    coords = offsets + grid
    # (b*c, h*w)| (b*c, h*w), (b*c, h*w, c)
    mapped_vals = th_batch_map_coordinates(input, coords)

    return mapped_vals
