""" @file rgb_prep.py You may run this script with arguments <path/to/rgb/image> <input_image_height> <input_image_width> to generate input images for the C model. """

import numpy as np
import sys
import os

# THE BELOW CODE HAS BEEN TAKEN FROM THE KERAS SOURCE CODE IN ORDER TO MAKE THE IMAGE PREPARATION PROSCESS FASTER AND LIGHTER
# THE CODE HAS BEEN TAKEN FROM keras.preprocessing.image

try:
  from PIL import Image as pil_image
except ImportError:
  pil_image = None

if pil_image is not None:
  _PIL_INTERPOLATION_METHODS = {
      'nearest': pil_image.NEAREST,
      'bilinear': pil_image.BILINEAR,
      'bicubic': pil_image.BICUBIC,
  }
  # These methods were only introduced in version 3.4.0 (2016).
  if hasattr(pil_image, 'HAMMING'):
    _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
  if hasattr(pil_image, 'BOX'):
    _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
  # This method is new in version 1.1.3 (2013).
  if hasattr(pil_image, 'LANCZOS'):
    _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def load_img(path, grayscale=False, target_size=None, interpolation='nearest'):
  """Loads an image into PIL format.
  Arguments:
      path: Path to image file
      grayscale: Boolean, whether to load the image as grayscale.
      target_size: Either `None` (default to original size)
          or tuple of ints `(img_height, img_width)`.
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are "nearest", "bilinear", and "bicubic".
          If PIL version 1.1.3 or newer is installed, "lanczos" is also
          supported. If PIL version 3.4.0 or newer is installed, "box" and
          "hamming" are also supported. By default, "nearest" is used.
  Returns:
      A PIL Image instance.
  Raises:
      ImportError: if PIL is not available.
      ValueError: if interpolation method is not supported.
  """
  if pil_image is None:
    raise ImportError('Could not import PIL.Image. '
                      'The use of `array_to_img` requires PIL.')
  img = pil_image.open(path)
  if grayscale:
    if img.mode != 'L':
      img = img.convert('L')
  else:
    if img.mode != 'RGB':
      img = img.convert('RGB')
  if target_size is not None:
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
      if interpolation not in _PIL_INTERPOLATION_METHODS:
        raise ValueError('Invalid interpolation method {} specified. Supported '
                         'methods are {}'.format(interpolation, ', '.join(
                             _PIL_INTERPOLATION_METHODS.keys())))
      resample = _PIL_INTERPOLATION_METHODS[interpolation]
      img = img.resize(width_height_tuple, resample)
  return img

def img_to_array(img, data_format=None):
  """Converts a PIL Image instance to a Numpy array.
  Arguments:
      img: PIL Image instance.
      data_format: Image data format.
  Returns:
      A 3D Numpy array.
  Raises:
      ValueError: if invalid `img` or `data_format` is passed.
  """
  if data_format is None:
    data_format = 'channels_last'
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ', data_format)
  # Numpy array x has format (height, width, channel)
  # or (channel, height, width)
  # but original PIL image has format (width, height, channel)
  x = np.asarray(img)
  if len(x.shape) == 3:
    if data_format == 'channels_first':
      x = x.transpose(2, 0, 1)
  elif len(x.shape) == 2:
    if data_format == 'channels_first':
      x = x.reshape((1, x.shape[0], x.shape[1]))
    else:
      x = x.reshape((x.shape[0], x.shape[1], 1))
  else:
    raise ValueError('Unsupported image shape: ', x.shape)
  return x

# UNTIL HERE

if not len(sys.argv)==4:
	print("Please run as RGB_to_txt.py <path to image> <target image height> <target image width>")
	sys.exit()

path_image = './input_images_txt/'
try:  
    os.mkdir(path_image)
except FileExistsError:
    pass



img = load_img(sys.argv[1], target_size = (int(sys.argv[2]), int(sys.argv[3])))
img = img_to_array(img, data_format='channels_first')
np.savetxt('{0}/{1}_0.txt'.format(path_image, os.path.basename(os.path.normpath(sys.argv[1])).split(".")[0]), img[0], fmt='%.8e')
np.savetxt('{0}/{1}_1.txt'.format(path_image, os.path.basename(os.path.normpath(sys.argv[1])).split(".")[0]), img[1], fmt='%.8e')
np.savetxt('{0}/{1}_2.txt'.format(path_image, os.path.basename(os.path.normpath(sys.argv[1])).split(".")[0]), img[2], fmt='%.8e')  
    