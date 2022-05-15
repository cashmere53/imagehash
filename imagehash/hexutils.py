# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from imagehash.container import ImageHash


def hex_to_hash(hexstr: str) -> ImageHash:
    """
    Convert a stored hash (hex, as retrieved from str(Imagehash))
    back to a Imagehash object.

    Args:
        hexstr (str): hex string

    Returns:
        ImageHash: hash

    Notes:
        1. This algorithm assumes all hashes are either
           bi-dimensional arrays with dimensions hash_size * hash_size,
           or onedimensional arrays with dimensions binbits * 14.
        2. This algorithm does not work for hash_size < 2.
    """
    hash_size: int = int(np.sqrt(len(hexstr) * 4))
    binary_array: str = f"{int(hexstr, 16):0>{hash_size * hash_size}b}"
    bit_rows: list[str] = [binary_array[i : i + hash_size] for i in range(0, len(binary_array), hash_size)]
    hash_array: npt.NDArray[np.bool_] = np.array([[bool(int(d)) for d in row] for row in bit_rows])
    return ImageHash(hash_array)


def hex_to_flathash(hexstr: str, hashsize: int) -> ImageHash:
    hash_size: int = int(len(hexstr) * 4 / (hashsize))
    binary_array: str = f"{int(hexstr, 16):0>{hash_size * hashsize}b}"
    hash_array: npt.NDArray[np.bool_] = np.array([[bool(int(d)) for d in binary_array]])[-hash_size * hashsize :]
    return ImageHash(hash_array)
