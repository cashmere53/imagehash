# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pywt
from PIL import ImageFilter
from PIL.Image import Resampling
from scipy import fftpack

from imagehash.container import ImageHash, ImageMultiHash

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Optional, TypeGuard

    import numpy.typing as npt
    from PIL.Image import Image

    AnyInt = int | np.int64
    ImgArr = npt.NDArray[np.uint8]
    BoolArr = npt.NDArray[np.bool_]
    FloatArr = npt.NDArray[np.float64]


def average_hash(image: Image, hash_size: int = 8, mean: Callable[..., Any] = np.mean) -> ImageHash:
    """
    Average Hash computation.

    Args:
        image (Image): must be a PIL instance.
        hash_size (int, optional): hash size. Defaults to 8.
        mean (Callable[..., Any], optional):
            how to determine the average luminescence. can try numpy.median instead. Defaults to np.mean.

    Returns:
        ImageHash: hash

    Notes:
        Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        Step by step explanation:
        https://web.archive.org/web/20171112054354/https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/
    """
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    # reduce size and complexity, then covert to grayscale
    image = image.convert("L").resize((hash_size, hash_size), Resampling.LANCZOS)

    # find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
    pixels: ImgArr = np.asarray(image)
    avg: np.float64 = mean(pixels)

    # create string of bits
    diff: BoolArr = pixels > avg
    # make a hash
    return ImageHash(diff)


def perceptual_hash(image: Image, hash_size: int = 8, highfreq_factor: int = 4) -> ImageHash:
    """
    Perceptual Hash computation.

    Args:
        image (Image): must be a PIL instance.
        hash_size (int, optional): hash size. Defaults to 8.
        highfreq_factor (int, optional): high frequency factor. Defaults to 4.

    Returns:
        ImageHash: perceptual hash

    Notes:
        Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    img_size: int = hash_size * highfreq_factor
    image = image.convert("L").resize((img_size, img_size), Resampling.LANCZOS)
    pixels: ImgArr = np.asarray(image)
    dct: FloatArr = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    dct_lowfreq: FloatArr = dct[:hash_size, :hash_size]
    med: np.float64 = np.median(dct_lowfreq)
    diff: BoolArr = dct_lowfreq > med
    return ImageHash(diff)


def perceptual_hash_simple(image: Image, hash_size: int = 8, highfreq_factor: int = 4) -> ImageHash:
    """
    Perceptual Hash computation.

    Args:
        image (Image): must be a PIL instance.
        hash_size (int, optional): hash size. Defaults to 8.
        highfreq_factor (int, optional): high frequency factor. Defaults to 4.

    Returns:
        ImageHash: perceptual hash

    Notes:
        Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    img_size: int = hash_size * highfreq_factor
    image = image.convert("L").resize((img_size, img_size), Resampling.LANCZOS)
    pixels: ImgArr = np.asarray(image)
    dct: FloatArr = fftpack.dct(pixels)
    dct_lowfreq: FloatArr = dct[:hash_size, 1 : hash_size + 1]
    avg: np.float64 = dct_lowfreq.mean()
    diff: BoolArr = dct_lowfreq > avg
    return ImageHash(diff)


def difference_hash(image: Image, hash_size: int = 8) -> ImageHash:
    """
    Difference Hash computation.
    computes differences horizontally

    Args:
        image (Image): must be a PIL instance.
        hash_size (int, optional): hash size. Defaults to 8.

    Raises:
        ValueError: Hash size must be greater than or equal to 2.

    Returns:
        ImageHash: difference hash

    Notes:
        following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    """
    # resize(w, h), but numpy.array((h, w))
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    image = image.convert("L").resize((hash_size + 1, hash_size), Resampling.LANCZOS)
    pixels: ImgArr = np.asarray(image)
    # compute differences between columns
    diff: BoolArr = pixels[:, 1:] > pixels[:, :-1]
    return ImageHash(diff)


def difference_hash_vertical(image: Image, hash_size: int = 8) -> ImageHash:
    """
    Difference Hash computation.
    computes differences vertically

    Args:
        image (Image): must be a PIL instance.
        hash_size (int, optional): hash size. Defaults to 8.

    Returns:
        ImageHash: difference hash vertical

    Notes:
        following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    """
    # resize(w, h), but numpy.array((h, w))
    image = image.convert("L").resize((hash_size, hash_size + 1), Resampling.LANCZOS)
    pixels: ImgArr = np.asarray(image)
    # compute differences between rows
    diff: BoolArr = pixels[1:, :] > pixels[:-1, :]
    return ImageHash(diff)


def wavelet_hash(
    image: Image,
    hash_size: int = 8,
    image_scale: Optional[int] = None,
    mode: Literal["haar", "db4"] = "haar",
    remove_max_haar_ll: bool = True,
) -> ImageHash:
    """
    Wavelet Hash computation.

    Args:
        image (Image): must be a PIL instance.
        hash_size (int, optional): must be a power of 2 and less than image_scale. Defaults to 8.
        image_scale (Optional[int], optional): must be power of 2 and less than image size.
            By default is equal to max power of 2 for an input image. Defaults to None.
        mode (Literal[&quot;haar&quot;, &quot;db4&quot;], optional): (see modes in pywt library). Defaults to "haar".
            'haar' - Haar wavelets, by default
            'db4' - Daubechies wavelets
        remove_max_haar_ll (bool, optional):
            remove the lowest low level (LL) frequency using Haar wavelet. Defaults to True.

    Raises:
        ValueError: wrong range

    Returns:
        ImageHash: image hash

    Notes:
        based on https://www.kaggle.com/c/avito-duplicate-ads-detection/
    """
    if image_scale is not None and (image_scale & (image_scale - 1)) == 0:
        raise ValueError("image_scale is not power of 2")
    else:
        image_natural_scale: int = 2 ** int(np.log2(min(image.size)))
        image_scale = max(image_natural_scale, hash_size)

    ll_max_level: int = int(np.log2(image_scale))
    level: int = int(np.log2(hash_size))
    if hash_size & (hash_size - 1) == 0:
        raise ValueError("hash_size is not power of 2")
    if level <= ll_max_level:
        raise ValueError("hash_size in a wrong range")

    dwt_level: int = ll_max_level - level

    image = image.convert("L").resize((image_scale, image_scale), Resampling.LANCZOS)
    pixels: FloatArr = np.asarray(image) / 255.0

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
    coeffs: list[FloatArr]
    if remove_max_haar_ll:
        coeffs = pywt.wavedec2(pixels, "haar", level=ll_max_level)
        coeffs = list(coeffs)
        coeffs[0] *= 0
        pixels = pywt.waverec2(coeffs, "haar")

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
    dwt_low: FloatArr = coeffs[0]

    # Substract median and compute hash
    med: np.float64 = np.median(dwt_low)
    diff: BoolArr = dwt_low > med
    return ImageHash(diff)


def color_hash(image: Image, binbits: int = 3) -> ImageHash:
    """
    Color Hash computation.

    Computes fractions of image in intensity, hue and saturation bins:
    * the first binbits encode the black fraction of the image
    * the next binbits encode the gray fraction of the remaining image (low saturation)
    * the next 6*binbits encode the fraction in 6 bins of saturation, for highly saturated parts of the remaining image
    * the next 6*binbits encode the fraction in 6 bins of saturation, for mildly saturated parts of the remaining image

    Args:
        image (Image): must be a PIL instance.
        binbits (int, optional): number of bits to use to encode each pixel fractions. Defaults to 3.

    Returns:
        ImageHash: color hash
    """
    # bin in hsv space:
    intensity: ImgArr = np.asarray(image.convert("L")).flatten()

    h: ImgArr
    s: ImgArr
    v: ImgArr
    h, s, v = [np.asarray(v).flatten() for v in image.convert("HSV").split()]

    # black bin
    mask_black: BoolArr = intensity < 256 // 8
    frac_black: np.float64 = mask_black.mean()

    # gray bin (low saturation, but not black)
    mask_gray: BoolArr = s < 256 // 3
    frac_gray: np.float64 = np.logical_and(~mask_black, mask_gray).mean()

    # two color bins (medium and high saturation, not in the two above)
    mask_colors: BoolArr = np.logical_and(~mask_black, ~mask_gray)
    mask_faint_colors: BoolArr = np.logical_and(mask_colors, s < 256 * 2 // 3)
    mask_bright_colors: BoolArr = np.logical_and(mask_colors, s > 256 * 2 // 3)

    c: AnyInt = max(1, mask_colors.sum())
    # in the color bins, make sub-bins by hue
    hue_bins: FloatArr = np.linspace(0, 255, 6 + 1)
    h_faint_counts: npt.NDArray[np.int64]
    if mask_faint_colors.any():
        h_faint_counts, _ = np.histogram(h[mask_faint_colors], bins=hue_bins)
    else:
        h_faint_counts = np.zeros(len(hue_bins) - 1, dtype=np.int64)

    h_bright_counts: npt.NDArray[np.int64]
    if mask_bright_colors.any():
        h_bright_counts, _ = np.histogram(h[mask_bright_colors], bins=hue_bins)
    else:
        h_bright_counts = np.zeros(len(hue_bins) - 1, dtype=np.int64)

    # now we have fractions in each category (6*2 + 2 = 14 bins)
    # convert to hash and discretize:
    max_value: int = 2**binbits
    values: list[int] = [
        min(max_value - 1, int(frac_black * max_value)),
        min(max_value - 1, int(frac_gray * max_value)),
    ]

    for counts in list(h_faint_counts) + list(h_bright_counts):
        values.append(min(max_value - 1, int(counts * max_value * 1.0 / c)))

    bitarray: list[bool] = []
    for val in values:
        bitarray += [val // (2 ** (binbits - i - 1)) % 2 ** (binbits - i) > 0 for i in range(binbits)]

    return ImageHash(np.asarray(bitarray).reshape((-1, binbits)))


def _tuple2_check(value: set[tuple[AnyInt, ...]]) -> TypeGuard[set[tuple[AnyInt, AnyInt]]]:
    ret: bool = True
    for v in value:
        if len(v) != 2:
            ret = False
    return ret


def _find_region(
    remaining_pixels: npt.NDArray[np.bool_], segmented_pixels: set[tuple[AnyInt, AnyInt]]
) -> set[tuple[AnyInt, AnyInt]]:
    """Finds a region and returns a set of pixel coordinates for it.

    Args:
        remaining_pixels (npt.NDArray[np.bool_]):
            A numpy bool array, with True meaning the pixels are remaining to segment
        segmented_pixels (set[tuple[int, int]]):
            A set of pixel coordinates which have already been assigned to segment. This will be
            updated with the new pixels added to the returned segment.

    Returns:
        set[tuple[np.int64, ...]]: region
    """
    in_region: set[tuple[AnyInt, ...]] = set()
    not_in_region: set[tuple[AnyInt, ...]] = set()
    # Find the first pixel in remaining_pixels with a value of True
    available_pixels: npt.NDArray[np.int64] = np.transpose(np.nonzero(remaining_pixels))
    start: tuple[np.int64, ...] = tuple(available_pixels[0])
    in_region.add(start)
    new_pixels: set[tuple[AnyInt, ...]] = in_region.copy()
    while True:
        try_next: set[tuple[AnyInt, AnyInt]] = set()

        # Find surrounding pixels
        for pixel in new_pixels:
            x, y = pixel
            neighbours: list[tuple[AnyInt, AnyInt]] = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            try_next.update(neighbours)

        # Remove pixels we have already seen
        try_next.difference_update(segmented_pixels, not_in_region)

        # If there's no more pixels to try, the region is complete
        if not try_next:
            break

        # Empty new pixels set, so we know whose neighbour's to check next time
        new_pixels = set()

        # Check new pixels
        for pixel in try_next:
            if remaining_pixels[pixel]:
                in_region.add(pixel)
                new_pixels.add(pixel)
                segmented_pixels.add(pixel)
            else:
                not_in_region.add(pixel)

    if not _tuple2_check(in_region):
        raise ValueError()

    return in_region


def _find_all_segments(
    pixels: npt.NDArray[np.float32], segment_threshold: int, min_segment_size: int
) -> list[set[tuple[AnyInt, AnyInt]]]:
    """
    Finds all the regions within an image pixel array, and returns a list of the regions.

    Args:
        pixels (npt.NDArray[np.float32]): A numpy array of the pixel brightnesses.
        segment_threshold (int): The brightness threshold to use when differentiating between hills and valleys.
        min_segment_size (int): The minimum number of pixels for a segment.

    Returns:
        list[set[tuple[AnyInt, AnyInt]]]: segments

    Notes:
        Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
        rounding in the greyscale conversion.
    """
    img_width, img_height = pixels.shape
    # threshold pixels
    threshold_pixels: npt.NDArray[np.bool_] = pixels > segment_threshold
    unassigned_pixels: npt.NDArray[np.bool_] = np.full(pixels.shape, True, dtype=bool)

    segments: list[set[tuple[AnyInt, AnyInt]]] = []
    already_segmented: set[tuple[AnyInt, AnyInt]] = set()

    # Add all the pixels around the border outside the image:
    already_segmented.update([(-1, z) for z in range(img_height)])
    already_segmented.update([(z, -1) for z in range(img_width)])
    already_segmented.update([(img_width, z) for z in range(img_height)])
    already_segmented.update([(z, img_height) for z in range(img_width)])

    # Find all the "hill" regions
    while np.bitwise_and(threshold_pixels, unassigned_pixels).any():
        remaining_pixels: npt.NDArray[np.bool_] = np.bitwise_and(threshold_pixels, unassigned_pixels)
        segment: set[tuple[AnyInt, AnyInt]] = _find_region(remaining_pixels, already_segmented)
        # Apply segment
        if len(segment) > min_segment_size:
            segments.append(segment)
        for pix in segment:
            unassigned_pixels[pix] = False

    print(f"{remaining_pixels=}, {type(remaining_pixels)=}")
    print(f"{segment=}, {type(segment)=}")

    # Invert the threshold matrix, and find "valleys"
    threshold_pixels_i = np.invert(threshold_pixels)
    while len(already_segmented) < img_width * img_height:
        remaining_pixels = np.bitwise_and(threshold_pixels_i, unassigned_pixels)
        segment = _find_region(remaining_pixels, already_segmented)
        # Apply segment
        if len(segment) > min_segment_size:
            segments.append(segment)
        for pix in segment:
            unassigned_pixels[pix] = False

    # print(locals())
    return segments


def crop_resistant_hash(
    image: Image,
    hash_func: Optional[Callable[..., ImageHash]] = None,
    limit_segments: Optional[int] = None,
    segment_threshold: int = 128,
    min_segment_size: int = 500,
    segmentation_image_size: int = 300,
) -> ImageMultiHash:
    """
    Creates a CropResistantHash object, by the algorithm described in the paper "Efficient Cropping-Resistant Robust
    Image Hashing". DOI 10.1109/ARES.2014.85

    This algorithm partitions the image into bright and dark segments, using a watershed-like algorithm, and then does
    an image hash on each segment. This makes the image much more resistant to cropping than other algorithms, with
    the paper claiming resistance to up to 50% cropping, while most other algorithms stop at about 5% cropping.

    Args:
        image (Image): The image to hash
        hash_func (Optional[Callable[..., ImageHash]], optional):
            The hashing function to use. Defaults to difference_hash.
        limit_segments (Optional[int], optional):
            If you have storage requirements, you can limit to hashing only the M largest segments. Defaults to None.
        segment_threshold (int, optional):
            Brightness threshold between hills and valleys. This should be static, putting it between
            peak and trough dynamically breaks the matching. Defaults to 128.
        min_segment_size (int, optional):
            Minimum number of pixels for a hashable segment. Defaults to 500.
        segmentation_image_size (int, optional):
            Size which the image is resized to before segmentation. Defaults to 300.

    Returns:
        ImageMultiHash: crop resistant hash

    Notes:
        Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
        rounding in the greyscale conversion. This leads to a slightly different result.
    """
    if hash_func is None:
        hash_func = difference_hash

    orig_image: Image = image.copy()
    # Convert to gray scale and resize
    image = image.convert("L").resize((segmentation_image_size, segmentation_image_size), Resampling.LANCZOS)
    # Add filters
    image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
    pixels = np.array(image).astype(np.float32)

    segments: list[set[tuple[AnyInt, AnyInt]]] = _find_all_segments(pixels, segment_threshold, min_segment_size)
    print(f"segments: {type(segments)=} {type(segments[0])=}")

    # If there are no segments, have 1 segment including the whole image
    if not segments:
        full_image_segment: set[tuple[AnyInt, AnyInt]] = {
            (0, 0),
            (segmentation_image_size - 1, segmentation_image_size - 1),
        }
        segments.append(full_image_segment)

    # If segment limit is set, discard the smaller segments
    if limit_segments is not None:
        segments = sorted(segments, key=lambda s: len(s), reverse=True)[:limit_segments]

    # Create bounding box for each segment
    hashes: list[ImageHash] = []
    for segment in segments:
        orig_w, orig_h = orig_image.size
        scale_w = float(orig_w) / segmentation_image_size
        scale_h = float(orig_h) / segmentation_image_size
        min_y = min(coord[0] for coord in segment) * scale_h
        min_x = min(coord[1] for coord in segment) * scale_w
        max_y = (max(coord[0] for coord in segment) + 1) * scale_h
        max_x = (max(coord[1] for coord in segment) + 1) * scale_w
        # Compute robust hash for each bounding box
        bounding_box: Image = orig_image.crop((min_x, min_y, max_x, max_y))
        hashes.append(hash_func(bounding_box))
        # Show bounding box
        # im_segment = image.copy()
        # for pix in segment:
        # 	im_segment.putpixel(pix[::-1], 255)
        # im_segment.show()
        # bounding_box.show()

    # print(locals())
    return ImageMultiHash(hashes)
