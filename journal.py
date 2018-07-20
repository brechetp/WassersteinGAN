__doc__ = """Journaling class"""

import os
import pdb
import utils
import math
import datetime
from tensorboardX import SummaryWriter
from IPython import embed
import numpy as np
import torch
import scipy as sp
import scipy.signal
import visualization as vs
import sqlite_utils as squ

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ROOT_PATH = os.getcwd()

class Journal(object):

    def __init__(self, opt):

        self.timestamp = now = datetime.datetime.now()

        dirnow = "{}{:02d}".format(months[now.month-1], now.day)
        strfnow = now.strftime('%H-%M-%S')
        name = strfnow

        self._writer = SummaryWriter(comment = dirnow + ' ' + name)
        self.opt = opt
        dirname = opt.outf

        self._root_path = os.path.realpath(ROOT_PATH)
        self._dir_path = os.path.join(self._root_path, dirname)
        self.opt.jpath = self._folder_path = os.path.join(self._root_path, dirname, dirnow, name)

        os.makedirs(self._folder_path, exist_ok=True)

        self._3dpoint_batch = dict()
        self._2dpoint_batch = dict()
        self._2dpoint= dict()
        self._matrix = dict()
        self._2dvector_batch = dict()
        self._matrix = dict()
        self._scalar= dict()
        self._histc = dict()

        return

    def add_data(self, key, val, n=None, batch=True):
        """Adds a vector to the journal"""
        val = val.detach().cpu()
        numel = val.numel()
        ndim = val.dim()
        if ndim == 0 or numel ==1:
            self._add_scalar(key, val.numpy(), n)
            return
        if batch:  # the first dimension is the batch dimension
            d = val.size(1)
            if d == 1:
                self._add_scalar(key, val.numpy(), n)  # correct?
            elif d == 2:
                self._add_2dpoint_batch(key, val.squeeze().numpy(), n)
            elif d == 3:
                self._add_3dpoint_batch(key, val.squeeze().numpy(), n)
            elif d == 4:  # quiver plot
                self._add_2dvector_batch(key, val.squeeze().numpy(), n)
            else:
                raise NotImplementedError('Dimension of data:', val.size())
            return
        else:
            ndim = val.dim()
            if ndim == 2:  # matrix
                self._add_matrix(key, val.numpy(), n)
            elif ndim == 1:  # vector
                d = val.size(0)
                if d == 2:
                    self._add_2dpoint(key, val.numpy(), n)
                elif d >= 3:
                    self._add_histc(key, val.numpy(), n)
            else:
                raise NotImplementedError()
        return

    def _add_matrix(self, key, val, n):
        """Val is a nbxnd value"""
        if not key in self._matrix.keys():
            self._matrix[key] = list()
        self._matrix[key].append((n, val))

    def _add_2dvector_batch(self, key, val, n=None):
        """Adds a 2dpoint_batch to the journal object

        :key: the ID of the vector
        :val: a tensor object for the tensor (nb x 4(xyuv))
        :n: the iteration number
        """
        if not key in self._2dvector_batch.keys():
            self._2dvector_batch[key] = list()
        self._2dvector_batch[key].append((n, val))
        return

    def _add_2dpoint(self, key, val, n):
        """Adds a 2dpoint_batch to the journal object

        :key: the ID of the vector
        :val: a tensor object for the tensor (nb x nd)
        :n: the iteration number
        """
        if not key in self._2dpoint.keys():
            self._2dpoint[key] = list()
        self._2dpoint[key].append((n, val))
        return

    def _add_2dpoint_batch(self, key, val, n=None):
        """Adds a 2dpoint_batch to the journal object

        :key: the ID of the vector
        :val: a tensor object for the tensor (nb x nd)
        :n: the iteration number
        """
        if n is None:
            self._add_2dpoint_batch_fixed(key, val)
        else:
            if not key in self._2dpoint_batch.keys():
                self._2dpoint_batch[key] = list()
            self._2dpoint_batch[key].append((n, val))
        return

    def _add_3dpoint_batch(self, key, val, n):
        """Adds a 3dpoint_batch to the journal object

        :key: the ID of the vector
        :val: a tensor object for the tensor (1D)
        :n: the iteration number
        """
        if not key in self._3dpoint_batch.keys():
            self._3dpoint_batch[key] = list()
        self._3dpoint_batch[key].append((n, val))
        return

    def _add_scalar(self, key, val, n):
        """Add a scalar to the journaling object"""
        self._writer.add_scalar(key, val, n)
        if not key in self._scalar.keys():
            self._scalar[key] = list()
        self._scalar[key].append((n, val))

        return

    def _log_db(self):
        '''logs the entry into the database'''

        conn = squ.create_table_opt(self.opt)
        with conn:
            n = squ.log_opt(conn, self.opt)
        return

    def close(self, *args, **kwargs):
        os.makedirs(self._folder_path, exist_ok=True)
        symlink = os.path.join(self._dir_path, 'latest')
        if os.path.exists(symlink):
            os.remove(symlink)
        os.symlink(self._folder_path, symlink)
        if self.opt is not None:
            utils.write_opt(self.opt, os.path.join(self._folder_path, 'config.txt'), self.timestamp)
        self._scatter_2d_samefig_batch(['generated', 'nu', 'mu'], bbox=utils.compute_2d_bbox(self._2dpoint_batch, ['nu', 'mu']), *args, **kwargs)
        self._plot_scalar(*args, **kwargs)
        self._scatter_2d(*args, **kwargs)
        # self._plot_matrix(*args, **kwargs)  to be defined
        self._plot_histc(nbins=100, *args, **kwargs)
        # self._scatter_2d_batch(*args, **kwargs)
        # self._scatter_3d_batch(*args, **kwargs)
        self._quiver_2d_batch(*args, **kwargs)
        self._plot_matrix(*args, **kwargs)
        self._log_db()

    def _plot_scalar(self, *args, **kargs):

        for key, rec in self._scalar.items():
            # for all key, val pairs
            tx = np.array(rec)
            t = tx[:, 0]
            x = sp.signal.medfilt(tx[:, 1])
            path = os.path.join(self._folder_path, key)
            filename = os.path.join(path, 'plot.png')
            os.makedirs(path, exist_ok=True)
            vs.plot_and_save(filename, t, x, label=key)

    def _plot_matrix(self, *args, **kargs):

        for key, rec in self._matrix.items():
            # for all key, val pairs
            path = os.path.join(self._folder_path, key)
            os.makedirs(path, exist_ok=True)
            lz = math.ceil(math.log10(1+rec[-1][0]))  # leading zeros
            arr = np.array([m for _,m in rec])
            vmin = arr.min()
            vmax = arr.max()
            for n, mat in rec:
                text = utils.textfiter(n, self.opt)
                filename = os.path.join(path, '{0:0>{1}d}'.format(n, lz))
                vs.imshow_and_save(filename, mat, label=key, vmin=vmin, vmax=vmax, text=text)

            if self.opt.movie:
                utils.make_movie(path, delete=self.opt.deletetmp)

    def _quiver_2d_batch(self, keys=None, *args, **kwargs):
        """Plot the 2d_batch vector"""
        if keys is None:
            keys = self._2dvector_batch.keys()
        for key, rec in self._2dvector_batch.items():
            if key not in keys:
                break
            # for all the values we collected as 3d_batch vectors
            path = os.path.join(self._folder_path, key)
            os.makedirs(path, exist_ok=True)
            # val is a dictionnary
            lz = math.ceil(math.log10(1+rec[-1][0]))
            for n, xyuv in rec: # the time (iteration)
                x = xyuv[:, 0]  # the origin for the quiver
                y = xyuv[:, 1]
                u = xyuv[:, 2]  # the delta x and y
                v = xyuv[:, 3]
                filename = os.path.join(path, '{0:0>{1}d}'.format(n, lz))
                vs.quiver_and_save(filename, X=x, Y=y, U=u, V=v, label=key)

    def _scatter_2d(self, *args, **kwargs):
        '''Scatter the data on the same plot'''

        for key, rec in self._2dpoint.items():

            lz = math.ceil(math.log10(1+rec[-1][0]))
            path = os.path.join(self._folder_path, key)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, 'plot.png')
            xy_data = np.array([r[1] for r in rec])
            text = "{} points".format(len(rec))
            z = np.array([r[0] for r in rec])  # the z coordinate
            vs.scatter_3d_and_save(filename, x=xy_data[:, 0], y=xy_data[:, 1], z=z, label=key, text=text, zlabel='niter')

    def _scatter_2d_batch(self, *args, **kwargs):
        """Plot the 2d_batch vector"""
        for key, rec in self._2dpoint_batch.items():
            # for all the values we collected as 3d_batch vectors
            # val is a dictionnary
            lz = math.ceil(math.log10(1+rec[-1][0]))
            path = os.path.join(self._folder_path, key)
            os.makedirs(path, exist_ok=True)
            for n, xy in rec: # the time (iteration)
                x = xy[:, 0]
                y = xy[:, 1]
                text = utils.textfiter(n, self.opt)
                filename = os.path.join(path, '{0:0>{1}d}'.format(n, lz))
                vs.scatter_and_save(filename, x, y, label=key, text=text)

        return

    def _scatter_2d_samefig_batch(self, keys=None, *args, **kwargs):
        '''Plot the different keys on the same figure'''
        if keys is None:
            keys = self._2dpoint_batch.keys()
        keys = [k for k in keys if k in self._2dpoint_batch.keys()]
        n_keys = len(keys)
        colors = kwargs.pop('colors', ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][:n_keys])
        markers = kwargs.pop('markers', ['+'])
        # pdb.set_trace()
        bbox = kwargs.pop('bbox', utils.compute_2d_bbox(self._2dpoint_batch, keys))
        # bbox is (x_min, x_max, y_min, y_max)
        num = len(self._2dpoint_batch[keys[0]])  # the number of iterations
        lz = math.ceil(math.log10(self._2dpoint_batch[keys[0]][-1][0]))
        lengths = np.array([len(self._2dpoint_batch[key]) for key in keys]) # the lengths of the collections
        fixed_dist = np.where(lengths==1)[0]  # the distributions that are fixed, we will not update them
        fixed_dist_names = [keys[i] for i in fixed_dist]  # the name of the fixed distributions

        # assert all(len(self._2dpoint_batch[key]) == num for key in keys), "The different collections are not of the same size"
        key_id = '_'.join([k[:min(len(k),3)] for k in keys])
        path = os.path.join(self._folder_path, key_id)
        os.makedirs(path, exist_ok=True)
        for i in range(num):  # for all the differente time steps
            x_min_bbox = min(i, bbox.shape[0]-1)
            xlim = list(bbox[x_min_bbox, 0:2])
            ylim = list(bbox[x_min_bbox, 2:])
            ax = None # will be changed in the loop
            for k, key in enumerate(keys):
                text=None  # overridden in the else condition (changing scatter)
                if key in fixed_dist_names: # fixed distribution plot, the same on every plots
                    _, xy = self._2dpoint_batch[key][0]
                    zorder = -1  # we plot the fixed distributions in the backgroud
                else:
                    n, xy = self._2dpoint_batch[key][i]
                    text = utils.textfiter(n, self.opt)
                    zorder = 1
                x = xy[:, 0]
                y = xy[:, 1]
                # pdb.set_trace()
                ax = vs.scatter_on_same_ax(x, y, ax=ax, zorder=zorder, color=colors[k], marker=markers[k%len(markers)], xlim=xlim, ylim=ylim, label=key, text=text)
            filename = os.path.join(path, '{0:0>{1}d}.png'.format(n, lz))  # the filename
            vs.save(filename)#, dpi=300)
        if self.opt.movie:
            utils.make_movie(path, delete=self.opt.deletetmp)
        return

    def _add_2dpoint_batch_fixed(self, key, val):
        '''Adds a fixed 2d_batch distribution'''
        self._2dpoint_batch[key] = [(0, val.reshape(val.shape[0], val.shape[1]))]  # won't be expanding

    def save_scatter(self, xy, name):
        """Plot the 2d_batch vector"""
        # for all the values we collected as 3d_batch vectors
        # val is a dictionnary
        x = xy[:, 0]
        y = xy[:, 1]
        path = os.path.join(self._folder_path, name)
        filename = os.path.join(path, 'scatter.png')
        os.makedirs(path, exist_ok=True)
        vs.scatter_and_save(filename, x, y, label=name)
        return

    def add_model(self, key, model, out, niter):
        """Saves the model and plots its graph"""
        self._writer.add_graph(model, out)
        torch.save(model.state_dict(), os.path.join(self._folder_path, '{}-i{:04d}.p'.format(key, niter)))
        return

    def _add_histc(self, key, val, niter):
        '''Stores the histograms of values'''
        if not key in self._histc.keys():
            self._histc[key] = []
        self._histc[key].append((niter, val))
        return

    def _scatter_3d_batch(self, *args, **kwargs):
        """Closes the writing object"""
        for key, rec in self._3dpoint_batch.items():
            # for all the values we collected as 3d_batch vectors
            # val is a dictionnary
            for n, xyz in rec:
                x = xyz[:, 0]
                y = xyz[:, 1]
                z = xyz[:, 2]
                path = (os.path.join(self._folder_path, '{}_{:04d}.png'.format(key, n)))
                vs.scatter_3d_and_save(filename, x, y, z, label=key)
        pass

    def _plot_histc(self, *args, **kwargs):

        for key, rec in self._histc.items():  # for all the histogram data
            lz = math.ceil(math.log10(1+rec[-1][0]))
            path = os.path.join(self._folder_path, key)
            os.makedirs(path, exist_ok=True)

            for n, hist in rec:
                filename = os.path.join(path, '{0:0>{1}d}.png'.format(n, lz))
                text = utils.textfiter(n, self.opt)
                vs.hist_and_save(filename, hist, label=key, text=text, **kwargs)

            if self.opt.movie:
                utils.make_movie(path, delete=self.opt.deletetmp)

    def _log_optim(self, key, optim, niter, field='cos'):
        '''Logs the optim field data as histogram'''

        for group in optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = optim.state[p]
                # pdb.set_trace()
                param_name = 'W{}x{}'.format(p.size(0), p.size(1)) if p.dim() == 2 else 'b{}'.format(p.size(0))
                key_data = '{}_{}-{}'.format(field, key, param_name)

                self.add_data(key_data, state[field], niter, batch=False)
        return

    def log_optim_grad(self, key, optim, niter):
        '''Logs the optim cosine data as histogram'''

        self._log_optim(key, optim, niter, field='grad')
        return

    def log_optim_cosine(self, key, optim, niter):
        '''Logs the optim cosine data as histogram'''

        self._log_optim(key, optim, niter, field='cos')
        return

    def log_optim_norm(self, key, optim, niter):
        '''Logs the optim norm data as histogram'''

        self._log_optim(key, optim, niter, field='norm')
        return

def main():
    journal = Journal()
    for n_iter in range(0, 100, 5):
        point3d = torch.randn(3)
        point2d = torch.randn(2)

        journal.add_data('A', point2d, n_iter)
        journal.add_data('B', point3d, n_iter)
    journal.close()

if __name__ == '__main__':
    main()
