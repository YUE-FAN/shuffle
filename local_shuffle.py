import torch
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.view(h // nrows, nrows, -1, ncols)
            .permute(0, 2, 1, 3).contiguous()
            .view(-1, nrows * ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.view(h//nrows, -1, nrows, ncols)
            .permute(0, 2, 1, 3).contiguous()
            .view(-1, ))

def forward(x, nrows, ncols):
    x_shape = x.size()  # [128, 3, 32, 32]
    x = x.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
    x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
    # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
    perm = torch.empty(0).float()
    for i in range(x_shape[1]):
        idx = torch.arange(x_shape[2] * x_shape[3]).view(x_shape[2], x_shape[3])
        idx = blockshaped(idx, nrows, ncols)
        for j in range(idx.size(0)):  # idx.size(0) is the number of blocks
            a = torch.randperm(nrows * ncols)
            idx[j] = idx[j][a]
        idx = idx.view(-1, nrows, ncols)
        idx = unblockshaped(idx, x_shape[2], x_shape[3]) + i * x_shape[2] * x_shape[3]
        perm = torch.cat((perm, idx.float()), 0)
    x_offset[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
    print(x_offset)
    return



a = torch.arange(16).view(1,1,4,4)

print(a)

b = forward(a,2,2)
