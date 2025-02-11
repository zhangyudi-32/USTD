"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

#####################################
# evaluation metrics
#####################################
def _rmse_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        rmse: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    rmse = np.sqrt((((y - label) ** 2) * valid_mask).sum() / (valid_count + 1e-7))

    return rmse


def _mae_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mae: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    mae = np.abs((y-label) * valid_mask).sum() / valid_count
    return mae

def _mape_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mape: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_mask = valid_mask * (np.abs(label) > 0.0001)
    valid_count = np.sum(valid_mask)

    mape = np.abs((y-label) / (label+1e-6) * valid_mask).sum() / valid_count
    return mape

def _quantile_CRPS_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [time, num_sample, num_m, dy]
        label: nd.array [time, num_m, dy]
        missing_index: [time, num_m, 1] or [time, num_m]
    Returns:
        CRPS: float
    """
    y = y.transpose(1, 0, 2, 3) # [num_sample, time, num_m, dy]
    def quantile_loss(target, forecast, q: float, eval_points) -> float:
        return 2 * np.sum(
            np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        )

    def calc_denominator(label, valid_mask):
        return np.sum(np.abs(label * valid_mask))

    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape[:2] == y.shape[:2]:
        missing_mask = missing_mask[:, :, np.newaxis]

    valid_mask = 1 - missing_mask
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(label, valid_mask)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = np.quantile(y, quantiles[i], axis=0)
        q_loss = quantile_loss(label, q_pred, quantiles[i], valid_mask)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)
    
def _quantile_CRPS_sum(y, label, missing_mask):
    label_sum = label.sum(axis=-1)[:,None]
    y_sum = y.sum(axis=-1)[:,None]
    missing_mask = missing_mask.mean(axis=-1)[:,None]
    return _quantile_CRPS_with_missing(y_sum, label_sum, missing_mask)

def _picp(y, all_gen_y, CI=95):
    low, high = (100 - CI) / 2, 100 - (100 - CI) / 2
    CI_y_pred = np.percentile(all_gen_y.cpu().numpy(), q=[low, high], axis=1)
    coverage = ((y.cpu().numpy() >= CI_y_pred[0]) & (y.cpu().numpy() <= CI_y_pred[1])).mean()
    return coverage

def _qice(y, all_gen_y, n_bins=5):
    quantile_list = np.linspace(0, 100, n_bins + 1)
    all_gen_y = all_gen_y.cpu().numpy().transpose(0, 2, 3, 1)
    y_pred_quantiles = np.percentile(all_gen_y, q=quantile_list, axis=-1).transpose(1, 2, 3, 0)
    quantile_membership = ((y.cpu().numpy()[..., None] - y_pred_quantiles) > 0).astype(int).sum(axis=-1)
    bin_counts = np.array([(quantile_membership == v).sum() for v in range(n_bins + 2)])
    bin_counts[1] += bin_counts[0]
    bin_counts[-2] += bin_counts[-1]
    bin_counts = bin_counts[1:-1]
    y_ratio = bin_counts / (y.shape[0] * y.shape[1] * y.shape[2])
    return np.abs(np.ones(n_bins) / n_bins - y_ratio).mean()
