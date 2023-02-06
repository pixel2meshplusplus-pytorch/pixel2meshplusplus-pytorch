import torch
import numpy as np

# Cameras helper functions

def normal(v):
    norm = torch.norm(v)
    if norm == 0:
        return v
    return torch.divide(v, norm)


def cameraMat(param):
    theta = param[0] * np.pi / 180.0
    camy = param[3] * torch.sin(param[1] * np.pi / 180.0)
    lens = param[3] * torch.cos(param[1] * np.pi / 180.0)
    camx = lens * torch.cos(theta)
    camz = lens * torch.sin(theta)
    Z = torch.stack([camx, camy, camz])

    x = camy * torch.cos(theta + np.pi)
    z = camy * torch.sin(theta + np.pi)
    Y = torch.stack([x, lens, z])
    X = torch.cross(Y, Z)

    cm_mat = torch.stack([normal(X), normal(Y), normal(Z)])
    return cm_mat, Z


def camera_trans(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    points = xyz[:, :3]
    pt_trans = points - o
    pt_trans = torch.matmul(pt_trans, torch.t(c))

    return pt_trans


def camera_trans_inv(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    inv_xyz = (torch.matmul(xyz, torch.linalg.inv(torch.t(c).float()))) + o
    return inv_xyz.half()


def gather_nd_torch(params, indices, batch_dim=0):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.
    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.
    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.
    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].
    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])
    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])
    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = torch.mean(input=x, dim=axis, keepdim=True)
    devs_squared = torch.square(x - m)
    return torch.mean(input=devs_squared, dim=axis, keepdim=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return torch.sqrt(reduce_var(x, axis=axis, keepdims=keepdims) + 1e-6)


def construct_feed_dict(pkl):

    stages=[pkl['stage1'],pkl['stage2'],pkl['stage3']]
    edges = []
    for i in range(1, 4):
        adj = pkl['stage{}'.format(i)][1]
        edges.append(adj[0])

    feed_dict = dict({'ellipsoid_feature_X': torch.tensor(pkl['coord'])})
    feed_dict.update({'edges': edges})
    feed_dict.update({'faces': pkl['faces']})
    feed_dict.update({'pool_idx': pkl['pool_idx']}) #2 pool_idx
    feed_dict.update({'lape_idx': pkl['lape_idx']})
    feed_dict.update({'supports': stages}) #3 supports
    feed_dict.update({'faces_triangle': pkl['faces_triangle']})#3 faces_traingles
    feed_dict.update({'sample_coord': pkl['sample_coord']})# (43,3) sample_coord for deformation hypothesis
    feed_dict.update({'sample_adj': pkl['sample_cheb_dense']})

    return feed_dict