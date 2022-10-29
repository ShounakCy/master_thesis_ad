import math
import logging
import numpy as np
import torch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import os

## Network parameters initialization
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='a',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


## Quintic spline definition.
def quintic_spline(x, z, a, b, c, d, e):
    return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5


## Fitting the trajectory of one planning circle by quintic spline, with the current location fixed.
def fitting_traj_by_qs(x, y):
    param, loss = curve_fit(quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [y[0]+1e-6, np.inf, np.inf, np.inf, np.inf, np.inf]))
    return param


## Custom activation for output layer (Graves, 2015)
def outputActivation(x, displacement=True):
    if displacement:
        # Then mu value denotes displacement.
        x[:, :, 0:2] = torch.stack([torch.sum(x[0:i, :, 0:2], dim=0) for i in range(1, x.shape[0] + 1)], 0)
    # Each output has 5 params to describe the gaussian distribution.
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)  # This positive value represents Reciprocal of SIGMA (1/sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)   # -1 < rho < 1
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho *
        torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) \
        + torch.log(torch.tensor(2 * math.pi))
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False, separately=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
                      - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
                      + torch.log(torch.tensor(2 * math.pi)))
                acc[:, :, count] = out + torch.log(wts)
                count += 1
        acc = -logsumexp(acc, dim=2)  # Negative log-likelihood
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc, dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = +(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
              - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
              + torch.log(torch.tensor(2 * math.pi)))
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc[:, :, 0]
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts


def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedMSETest(y_pred, y_gt, mask, separately=False):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
              
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts


## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    # Get the maximal probability value from 6 full path
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    # here (inputs - s) is to compare the relative probability with the most probable behavior.
    # and then sum up all candidate behaviors.
    # s->logP(Y | m_max,X), inputs->logP(m_i,Y | X), (inputs - s)->logP(m_i | X)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

# def make_segments(x, y):
#     '''
#     Create list of line segments from x and y coordinates, in the correct format for LineCollection:
#     an array of the form   numlines x (points per line) x 2 (x and y) array
#     '''

#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
#     return segments

# def colorline(x, y, z=None, cmap=plt.get_cmap('rainbow'), norm=plt.Normalize(0.0, 1.0), linewidth=4, alpha=0.9, zorder=3):
#       """
#       http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
#       http://matplotlib.org/examples/pylab_examples/multicolored_line.html
#       Plot a colored line with coordinates x and y
#       Optionally specify colors in the array z
#       Optionally specify a colormap, a norm function and a line width
#       """
#       # Default colors equally spaced on [0,1]:
#       if z is None:
#           z = np.linspace(0.0, 1.0, len(x))
#       z = np.asarray(z)
#       segments = make_segments(x, y)
#       lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, zorder= zorder,
#                                 linewidth=linewidth, alpha=alpha)
#       ax =plt.gca()
#       ax.add_collection(lc)
#       return lc