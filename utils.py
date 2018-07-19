import os
import torch.utils.data
from IPython import embed
import pdb
import torch
import numpy as np
import subprocess
import tempfile
import shutil
import argparse
import sys
import glob

INFTY = 10**15  # infinity
EPS = 10**(-2)  # small value (?)

class OverfitSampler(torch.utils.data.sampler.Sampler):
    """Sampler on the dataset to overfit"""

    def __init__(self, n_samples):
        assert n_samples > 0
        self._n_samples = n_samples
    def __iter__(self):
        return iter(range(self._n_samples))
    def __len__(self):
        return self._n_samples

def write(path, **kwargs):
    """Writes the options to the file

    Assumes opt has a __print__ method"""
    with open(path, 'w') as _file:
        for key, arg in kwargs.items():
            _file.write("--- {} ---\n\n".format(key))
            _file.write(str(arg) + '\n\n')
    return

def write_opt(opt, path, timestamp=None):
    """Writes the options to the file

    Assumes opt has a __print__ method"""
    with open(path, 'w') as _file:
        if timestamp is not None:
            _file.write(timestamp.strftime('%Y-%m-%d %H:%M:%S\n'))
        for key, arg in opt.__dict__.items():
            # if key != 'config':
            _file.write("{}: {}\n".format(key, arg))
    return

def textfiter(n, opt):
    '''formats the string for the iter number '''
    ne = n // (opt.epochSize) + 1
    nb = n %  (opt.epochSize // (opt.Diters + 1)) + 1
    text = "e:{:0>2d} b:{:0>2d} (n:{:d})".format(ne, nb, n)
    return  text

def max_or_first(max_val, arr, d, axis=1):
        return np.maximum(max_val, arr[..., d].max(axis=axis)) if max_val is not None else arr[..., d].max(axis=axis)
def min_or_first(min_val, arr, d, axis=1):
        return np.minimum(min_val, arr[..., d].min(axis=axis)) if min_val is not None else arr[..., d].min(axis=axis)

def compute_2d_bbox(d, keys, aspect='equal'):
    max_x, min_x, max_y, min_y = None, None, None, None
    # pdb.set_trace()
    for key in keys:
        val = np.array([v for _,v in d[key]])
        max_x = max_or_first(max_x, val, d=0)
        min_x = min_or_first(min_x, val, d=0)
        max_y = max_or_first(max_y, val, d=1)
        min_y = min_or_first(min_y, val, d=1)
    bbox = np.zeros((max_x.shape[0], 4))
    bbox[:, 0] = min_x
    bbox[:, 1] = max_x
    bbox[:, 2] = min_y
    bbox[:, 3] = max_y
    if aspect == 'equal':
        ratio = (max_y - min_y) / (max_x - min_x)  # for every frame, the ratio
        # if the ratio is over 1, we have to diminish it i.e. inscrease max_x - min_x
        max_x[ratio>1] = 1/2 * (ratio*(max_x-min_x) + (max_x + min_x))[ratio>1]
        min_x[ratio>1] = (bbox[..., 1] + min_x- max_x)[ratio>1]
        max_y[ratio<1] = 1/2 * (1/ratio*(max_y-min_y) + (max_y + min_y))[ratio<1]
        min_y[ratio<1] = (bbox[..., 3] + min_y- max_y)[ratio<1]
        bbox[:, 0] = min_x
        bbox[:, 1] = max_x
        bbox[:, 2] = min_y
        bbox[:, 3] = max_y
    return bbox

def make_movie(path: str, delete: bool = False):
    '''Makes a movie from pictures inside a folder'''

    path = os.path.realpath(path)
    oldwd = os.getcwd()
    os.chdir(path)
    basename = os.path.basename(path)
    fname = os.path.join('/'.join(path.split('/')[:-1]), basename + '.mp4')  # the name of the video
    list_files = glob.glob('*.png')
    fname_flist = os.path.join(path, 'flist.txt')
    with open(fname_flist, 'w') as file_flist:
        for file in sorted(list_files):
            file_flist.write('file {}\n'.format(file))

    try:
        ffmpeg = subprocess.check_call(['ffmpeg', '-y', '-r', '12', '-f', 'concat', '-re', '-i', fname_flist, '-c:v', 'libx264',  '-pix_fmt', 'yuv420p', fname])
        if delete:  # we delete the individual frames
            shutil.rmtree(path)
    except:
        print('Error creating the video in ', path)

    os.chdir(oldwd)
    return

def filter_kwargs(kw: dict, fun,  *args, **kwargs):
    # target = user_defined
    target = set(fun.__code__.co_varnames)  # the target dictionary
    if len(args) > 0:  # more functions to take into consideration
        for f in args:
            target.update(f.__code__.co_varnames)
    add_kwargs = kwargs.pop('add_kwargs', {})
    target.update(add_kwargs)
    return {k: kw[k] for k in kw.keys() if k in target}

def optsffile(filename, namespace):
    '''Returns a opt string configuration file from a filename'''
    opt_lst = []
    keys = namespace.__dict__.keys()  # the allowed keys for the arguments
    with open(filename, 'r') as file:
        for line in file:
            record = line.strip().split(': ')  # assumes this partition in the config file
            if record[0] in keys:
                keys -= record[0]
                opt_lst += parse_record(record)
    return opt_lst

def parse_record(record):
    '''Parse the list of fields'''
    out = []
    if len(record) == 2:  # name of the record and its value
        f1, f2 = record  # two fields
        if f2 == 'False':  # assumes all the boolean flags are store_true
            return out
        else:
            out = ["--{}".format(f1)]
            if not f2 == 'True': # else we leave the field blank, assuming store_true behaviour
                # out += [tr_field(f2)]
                f2 = f2.lstrip('[').rstrip(']').split(',')  # strip the list fields [..., ...]
                if f2 != ['']:
                    out += f2
            return out
    return out

def tr_field(f):
    '''Checks if digit or str field...'''
    try:
        return int(f)
    except ValueError:  # if not int
        try:
            return float(f)
        except ValueError:
            return f

class LoadFromFile(argparse.Action):
    '''Class to load a namespace from a file'''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)  # super constructor

    def __call__(self, parser, namespace, values, option_string=None):
        '''Will be called by the parser upon reading a file'''
        if values == '1':  # trick to flag a configuration read in a file
            namespace.config = '1'
            return
        elif values ==  'None':
            return  # default value for the config file, do nothing
        else:
            if not os.path.isfile(values):
                raise ValueError('The argument is not a file', values)
            opt_lst = optsffile(values, namespace)
            opt_lst.extend(parse_record(['config', '1']))
            parser.parse_args(opt_lst, namespace=namespace)
        return

def fetch_2d_gradient_W1(model):
    '''Returns the quiver data for the gradient of the 2D first layer'''

    for key, val in model.main._modules.items():
        if key.find('Linear1') != -1:
            return torch.cat((val.weight.data, val.weight.grad), dim=1)

def test(args):
    make_movie(args[1])

if __name__ == "__main__":
    test(sys.argv)
