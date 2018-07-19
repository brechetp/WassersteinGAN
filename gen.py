__doc__ = """Synthetic data generation for PyTorch use"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import pdb
from IPython import embed
import utils
import math
import numpy.random
import random

def mm_gaussian(nsample, means, covars, weights):
    """Generates a mixture model of gaussians

    :ngaussian: the number of gaussians
    :nsample: the number of samples
    :nd: the dimension for the gaussian
    :means: the list of means
    :covar: the list of covariance matrices
    :weights: the weights for each of the gaussian

    :return: the samples in tensor format
    """
    assert len(means) == len(covars), "Number of means or covariance matrices inconsistant with the number of gaussians"
    ngaussian = len(means)
    nd = means[0].size(0)
    weights.div_(weights.sum())
    #  weights = torch.tensor([0.5, 0.5])
    #  means = torch.tensor([[-3, 0], [3, 0]], dtype=torch.float)
    samples = torch.zeros(ngaussian, nsample, nd)
    for i, (mean, covar) in enumerate(zip(means, covars)):
        #  covar = I
        #  covar.div_(covar.max())
        #  corr = 0.01 * (R.t() + R) + 3*I  # cross correletion matrix
        #  covar = corr - torch.mm(mean.unsqueeze(1), mean.unsqueeze(1).t())
        multi_normal = MultivariateNormal(loc=mean, covariance_matrix=covar)
        samples[i] = multi_normal.sample((nsample,))
    indices = np.random.permutation(nsample)  # the total range of indices
    range_idx = (0, 0)
    mm_sample = samples[0]  # the mixture model for the gaussian
    for i in range(ngaussian):
        n = int(0.5 + weights[i] * nsample)  # the number of samples belonging to this
        range_idx = range_idx[1], min(n+range_idx[1], nsample)
        idx = indices[range_idx[0]:range_idx[1]]
        mm_sample[idx] = samples[i, idx]
    return mm_sample.unsqueeze(2).unsqueeze(3)

def rotation_matrix(nd, theta):
    R = torch.eye(3, 3, dtype=torch.float)
    R[0, 0], R[0, 1] = np.cos(theta), -np.sin(theta)
    R[1, 0], R[1, 1] = np.sin(theta), np.cos(theta)
    return R[:nd, :nd]

def random_mc(nd, ngaussian, random=True):
    '''Random parameters for the gaussians'''
    assert nd > 0 and ngaussian > 0
    theta = torch.zeros(1)
    R = torch.zeros(nd, nd)
    eigenvalues = torch.zeros((nd,))
    mean = torch.zeros(nd)
    det = 0.01
    covars = ngaussian * [0]
    means = ngaussian * [0]
    e = torch.zeros(nd)
    scale = 10
    e[0] = scale  # first vector
    origin = torch.zeros(nd)
    rot = 0
    for i in range(ngaussian):
        R.uniform_(0, 1)
        theta.uniform_(0, 2*math.pi)
        rot += (2 * math.pi / ngaussian)
        if random:
            mean.uniform_(-scale, scale)
        else:
            mean = origin + rotation_matrix(nd, rot) @ e
        eigenvalues.random_(1, to=10)
        eigenvalues.div_(eigenvalues.max())
        # pdb.set_trace()
        U = rotation_matrix(nd, theta)
        D = torch.diag(eigenvalues)
        covars[i] = (0.001 * det * (R + R.t()) + U @ D @ U.t()).clone()
        means[i] = mean.clone()
    weights = torch.randint(1, ngaussian+1, size=(ngaussian,))
    return means, covars, weights

G_COVARS = 3 * [0]
G_MEANS = 3 * [0]

torch.manual_seed(2)
np.random.seed(2)

for nd in range(1, 4):
    means, covars, _ = random_mc(nd, 25)
    G_COVARS[nd-1] = covars
    G_MEANS[nd-1] = means

def get_means_covars(nd, ngaussian, random):
    #  global G_COVARS, G_MEANS
    #  if random:
    #      return random_mc(nd, ngaussian)
    #  else:
    #      covars = G_COVARS[nd-1][:ngaussian]
    #      means = G_MEANS[nd-1][:ngaussian]
    #      weights = torch.ones(ngaussian)
        #  return means, covars, weights
    return random_mc(nd, ngaussian, random)

def gaussian_dataset(ngaussian, nd, nsample, random=True):
    """The function called by the script"""
    means, covars, weights = get_means_covars(nd, ngaussian, random)
    return mm_gaussian(nsample, means, covars, weights)

def gaussian_gen(ngaussian, nd, batchSize, seed, random=True):
    """Function called to yield infinity values"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    means, covars, weights = get_means_covars(nd, ngaussian, random)
    while True:
        # yield multi_normal.sample((batchSize,))
        yield mm_gaussian(batchSize, means, covars, weights)

def manual_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():

    embed()

if __name__ == '__main__':
    main()
