'''Image I/O and preprocessing, ported from caffe/python/caffe/io.py (BSD 2-Clause, BVLC).'''

import numpy as np
import skimage
import skimage.io
from skimage.transform import resize
from scipy.ndimage import zoom

from .net import blobproto_to_array   # noqa: F401  (re-exported like caffe.io.blobproto_to_array)


class Transformer(object):
    '''Transform input for feeding into a Net (see caffe.io.Transformer).'''

    def __init__(self, inputs):
        self.inputs = inputs
        self.transpose = {}
        self.channel_swap = {}
        self.raw_scale = {}
        self.mean = {}
        self.input_scale = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception('{} is not one of the net inputs: {}'.format(in_, self.inputs))

    def preprocess(self, in_, data):
        '''(H x W x K) image -> (K x H x W) net input: convert to single,
        resize, transpose, reorder channels, scale raw input, subtract
        mean, scale feature.'''
        self.__check_input(in_)
        caffe_in = data.astype(np.float32, copy=True)
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        in_dims = tuple(self.inputs[in_][2:])
        if caffe_in.shape[:2] != in_dims:
            caffe_in = resize_image(caffe_in, in_dims)
        if transpose is not None:
            caffe_in = caffe_in.transpose(transpose)
        if channel_swap is not None:
            caffe_in = caffe_in[list(channel_swap), :, :]
        if raw_scale is not None:
            caffe_in *= raw_scale
        if mean is not None:
            caffe_in -= mean
        if input_scale is not None:
            caffe_in *= input_scale
        return caffe_in

    def deprocess(self, in_, data):
        '''Invert Caffe formatting; see preprocess().'''
        self.__check_input(in_)
        decaf_in = data.copy().squeeze()
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        if input_scale is not None:
            decaf_in /= input_scale
        if mean is not None:
            decaf_in += mean
        if raw_scale is not None:
            decaf_in /= raw_scale
        if channel_swap is not None:
            decaf_in = decaf_in[np.argsort(channel_swap), :, :]
        if transpose is not None:
            decaf_in = decaf_in.transpose(np.argsort(transpose))
        return decaf_in

    def set_transpose(self, in_, order):
        self.__check_input(in_)
        if len(order) != len(self.inputs[in_]) - 1:
            raise Exception('Transpose order needs to have the same number of dimensions as the input.')
        self.transpose[in_] = order

    def set_channel_swap(self, in_, order):
        self.__check_input(in_)
        if len(order) != self.inputs[in_][1]:
            raise Exception('Channel swap needs to have the same number of dimensions as the input channels.')
        self.channel_swap[in_] = order

    def set_raw_scale(self, in_, scale):
        self.__check_input(in_)
        self.raw_scale[in_] = scale

    def set_mean(self, in_, mean):
        self.__check_input(in_)
        mean = np.asarray(mean, dtype=np.float32)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
            if ms != tuple(self.inputs[in_][1:]):
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean[in_] = mean

    def set_input_scale(self, in_, scale):
        self.__check_input(in_)
        self.input_scale[in_] = scale


def load_image(filename, color=True):
    '''Load an image as np.float32 in [0, 1], (H x W x 3) RGB or (H x W x 1) grayscale.'''
    img = skimage.io.imread(filename, as_gray=not color)
    img = skimage.img_as_float(img).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):
    '''Resize an (H x W x K) image to new_dims = (height, width) with interpolation.'''
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, tuple(new_dims), order=interp_order, mode='constant',
                                 anti_aliasing=False)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]), dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


def oversample(images, crop_dims):
    '''Crop images into the four corners, center, and their mirrored versions.'''
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([-crop_dims / 2.0, crop_dims / 2.0])
    crops_ix = np.tile(crops_ix, (2, 1))

    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1], im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix - 5:ix] = crops[ix - 5:ix, :, ::-1, :]  # flip for mirrors
    return crops
