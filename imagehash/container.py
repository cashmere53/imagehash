# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import numpy.typing as npt


def _binary_array_to_hex(arr: npt.NDArray[np.bool_]) -> str:
    """
    internal function to make a hex string out of a binary array.
    """

    bit_string: str = "".join(str(b) for b in 1 * arr.flatten())
    width: int = int(np.ceil(len(bit_string) / 4))

    return f"{int(bit_string, 2):0>{width}x}"


class ImageHash:
    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """

    def __init__(self, binary_array: npt.NDArray[np.bool_]) -> None:
        self.hash: npt.NDArray[np.bool_] = binary_array

    def __str__(self) -> str:
        return _binary_array_to_hex(self.hash.flatten())

    def __repr__(self) -> str:
        return repr(self.hash)

    def __sub__(self, other: object) -> int:
        if other is None:
            raise TypeError("Other hash must not be None.")
        if not isinstance(other, ImageHash):
            return NotImplemented
        if self.hash.size != other.hash.size:
            raise TypeError(
                "ImageHashes must be of the same shape.",
                self.hash.shape,
                other.hash.shape,
            )

        nonzero_count: int | Any = np.count_nonzero(self.hash.flatten(), other.hash.flatten())
        if not isinstance(nonzero_count, int):
            raise ValueError()

        return nonzero_count

    def __hash__(self) -> int:
        """this returns a 8 bit integer, intentionally shortening the information"""
        return sum([2 ** (i % 8) for i, v in enumerate(self.hash.flatten()) if v])

    def __len__(self) -> int:
        """Returns the bit length of the hash"""
        return self.hash.size


class DiffResult(NamedTuple):
    length: int
    dist_sum: int


class ImageMultiHash:
    """
    This is an image hash containing a list of individual hashes for segments of the image.
    The matching logic is implemented as described in Efficient Cropping-Resistant Robust Image Hashing
    """

    def __init__(self, hashes: list[ImageHash]) -> None:
        self.segment_hashes: list[ImageHash] = hashes

    def __eq__(self, other: Optional[object]) -> bool:
        if other is None:
            return False
        if not isinstance(other, ImageMultiHash):
            return NotImplemented

        return self.matches(other)

    def __ne__(self, other: Optional[object]) -> bool:
        if not isinstance(other, ImageMultiHash):
            return NotImplemented

        return not self.matches(other)

    def __sub__(
        self, other: object, hamming_cutoff: Optional[float] = None, bit_error_rate: Optional[float] = None
    ) -> float:
        if not isinstance(other, ImageMultiHash):
            return NotImplemented

        matches, sum_distance = self.hash_diff(other, hamming_cutoff, bit_error_rate)
        max_difference = len(self.segment_hashes)
        if matches == 0:
            return max_difference

        max_distance = matches * len(self.segment_hashes[0])
        tie_breaker = 0 - (float(sum_distance) / max_distance)
        match_score = matches + tie_breaker
        return float(max_difference - match_score)

    def __hash__(self) -> int:
        return hash(tuple(hash(segment) for segment in self.segment_hashes))

    def __str__(self) -> str:
        return ",".join(str(x) for x in self.segment_hashes)

    def __repr__(self) -> str:
        return repr(self.segment_hashes)

    def hash_diff(
        self, other_hash: ImageMultiHash, hamming_cutoff: Optional[float] = None, bit_error_rate: Optional[float] = None
    ) -> DiffResult:
        """
        Gets the difference between two multi-hashes, as a tuple. The first element of the tuple is the number of
        matching segments, and the second element is the sum of the hamming distances of matching hashes.

        Args:
            other_hash (ImageMultiHash): The image multi hash to compare against
            hamming_cutoff (Optional[float], optional):
                The maximum hamming distance to a region hash in the target hash. Defaults to None.
            bit_error_rate (Optional[float], optional):
                Percentage of bits which can be incorrect, an alternative to the hamming cutoff.
                The default of 0.25 means that the segment hashes can be up to 25% different.

        Raises:
            ValueError: need hamming_cutoff param.

        Returns:
            tuple[int, int]: distances length and sum of distances

        Notes:
            Do not order directly by this tuple, as higher is better for matches, and worse for hamming cutoff.
        """
        # Set default hamming cutoff if it's not set.
        if hamming_cutoff is None and bit_error_rate is None:
            bit_error_rate = 0.25
        if hamming_cutoff is None and bit_error_rate is not None:
            hamming_cutoff = len(self.segment_hashes[0]) * bit_error_rate
        if hamming_cutoff is None:
            raise ValueError("need hamming_cutoff param.")

        # Get the hash distance for each region hash within cutoff
        distances: list[int] = []
        for segment_hash in self.segment_hashes:
            lowest_distance: int = min(
                segment_hash - other_segment_hash for other_segment_hash in other_hash.segment_hashes
            )
            if lowest_distance > hamming_cutoff:
                continue
            distances.append(lowest_distance)
        # return len(distances), sum(distances)
        return DiffResult(len(distances), sum(distances))

    def matches(
        self,
        other_hash: ImageMultiHash,
        region_cutoff: int = 1,
        hamming_cutoff: Optional[float] = None,
        bit_error_rate: Optional[float] = None,
    ) -> bool:
        """Checks whether this hash matches another crop resistant hash, `other_hash`.

        Args:
            other_hash (ImageMultiHash): The image multi hash to compare against
            region_cutoff (int, optional): The minimum number of regions which must have a matching hash. Defaults to 1.
            hamming_cutoff (Optional[float], optional):
                The maximum hamming distance to a region hash in the target hash. Defaults to None.
            bit_error_rate (Optional[float], optional):
                Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
                default of 0.25 means that the segment hashes can be up to 25% different.

        Returns:
            bool: match result
        """
        matches, _ = self.hash_diff(other_hash, hamming_cutoff, bit_error_rate)
        return matches >= region_cutoff

    def best_match(
        self,
        other_hashes: Sequence[ImageMultiHash],
        hamming_cutoff: Optional[float] = None,
        bit_error_rate: Optional[float] = None,
    ) -> ImageMultiHash:
        """Returns the hash in a list which is the best match to the current hash

        Args:
            other_hashes (Sequence[ImageMultiHash]): A list of image multi hashes to compare against
            hamming_cutoff (Optional[float], optional):
                The maximum hamming distance to a region hash in the target hash. Defaults to None.
            bit_error_rate (Optional[float], optional):
                Percentage of bits which can be incorrect, an alternative to the hamming cutoff.
                Defaults to 0.25 if unset, which means the hash can be 25% different.

        Returns:
            ImageMultiHash: image multi hash
        """
        return min(other_hashes, key=lambda other_hash: self.__sub__(other_hash, hamming_cutoff, bit_error_rate))
