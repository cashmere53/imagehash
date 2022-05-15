# -*- coding: utf-8 -*-

"""
Image hashing library
======================

Example:

>>> from PIL import Image
>>> import imagehash
>>> hash = imagehash.average_hash(Image.open('test.png'))
>>> print(hash)
d879f8f89b1bbf
>>> otherhash = imagehash.average_hash(Image.open('other.bmp'))
>>> print(otherhash)
ffff3720200ffff
>>> print(hash == otherhash)
False
>>> print(hash - otherhash)
36
>>> for r in range(1, 30, 5):
...     rothash = imagehash.average_hash(Image.open('test.png').rotate(r))
...     print('Rotation by %d: %d Hamming difference' % (r, hash - rothash))
...
Rotation by 1: 2 Hamming difference
Rotation by 6: 11 Hamming difference
Rotation by 11: 13 Hamming difference
Rotation by 16: 17 Hamming difference
Rotation by 21: 19 Hamming difference
Rotation by 26: 21 Hamming difference
>>>
"""

from imagehash.container import ImageHash, ImageMultiHash
from imagehash.hashfunc import (
    average_hash,
    color_hash,
    crop_resistant_hash,
    difference_hash,
    difference_hash_vertical,
    perceptual_hash,
    perceptual_hash_simple,
    wavelet_hash,
)
from imagehash.hexutils import hex_to_flathash, hex_to_hash

__version__ = "0.1.0"

__all__ = [
    "ImageHash",
    "ImageMultiHash",
    "average_hash",
    "color_hash",
    "crop_resistant_hash",
    "difference_hash",
    "difference_hash_vertical",
    "perceptual_hash",
    "perceptual_hash_simple",
    "wavelet_hash",
    "hex_to_hash",
    "hex_to_flathash",
]
