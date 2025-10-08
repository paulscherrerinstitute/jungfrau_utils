import h5py
import numpy as np
import pytest

from jungfrau_utils import File
from tests.const_JF01T03V01 import *


def calc_downsample(downsample):

    ds_shape = (
        (DATA_SHAPE[0] + downsample[0] - 1) // downsample[0],
        (DATA_SHAPE[1] + downsample[1] - 1) // downsample[1],
    )
    ds = np.zeros((STACK_SHAPE[0], *ds_shape), dtype=np.float32)

    if downsample == (DATA_SHAPE[0], DATA_SHAPE[1]):
        ds_px_count = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    else:
        ds_px_count = downsample[0] * downsample[1]
    for j in range(ds_shape[0]):
        i_y = downsample[0] * j
        rng_y = slice(i_y, min(i_y + downsample[0], DATA_SHAPE[0]))
        for k in range(ds_shape[1]):
            i_x = downsample[1] * k
            rng_x = slice(i_x, min(i_x + downsample[1], DATA_SHAPE[1]))
            gpr = np.count_nonzero(inv_pixel_mask[rng_y, rng_x]) / ds_px_count
            ds[:, j, k] = (
                np.sum(converted_image_stack_mask[:, rng_y, rng_x], axis=(1, 2)) / gpr if gpr else 0
            )
    return ds


def test_file_adapter(file_adapter, gain_file, pedestal_file):
    assert file_adapter.gain_file == gain_file
    assert file_adapter.pedestal_file == pedestal_file


@pytest.mark.parametrize(
    "idx,dim,shape",
    [
        ((0,), (2, 2, 2), (IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_SHAPE)),
        ((slice(None, 3),), (3, 3, 3), (STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE)),
        (([0, 2, 4],), (3, 3, 3), (STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE)),
        (((0, 2, 4),), (3, 3, 3), (STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE)),
        ((0, 2, 4), (0,), ((),)),  # a special case, but has the same behaviour as h5py
        ((range(0, 5, 2),), (3, 3, 3), (STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE, STACK_IMAGE_SHAPE)),
    ],
)
def test_file_get_part_image(file_adapter, idx, dim, shape):
    for i in range(len(dim)):
        res = file_adapter[idx + (slice(None),) * i]

        assert res.dtype == np.dtype(np.float32)
        assert res.ndim == dim[i]
        assert res.shape == shape[i]

        if dim[i] == 0:
            continue

        # only corners tested as there is no geom/gap correction
        for sub_idx in [
            (slice(None, 256), slice(None, 256)),
            (slice(None, 256), slice(-256, None)),
            (slice(-256, None), slice(None, 256)),
            (slice(-256, None), slice(-256, None)),
        ]:
            idx_res = (slice(None),) * (dim[i] - 2) + sub_idx
            idx_mask = idx * (dim[i] - 2) + sub_idx
            test_res = (
                converted_image_single_mask[idx_mask]
                if dim[i] == 2
                else converted_image_stack_mask[idx_mask]
            )
            assert np.allclose(res[idx_res], test_res)


@pytest.mark.parametrize(
    "idx",
    [
        slice(None),
        [0, 2, 4],
        (0, 2, 4),
        range(0, 5, 2),
    ],
)
def test_file_export(file_adapter, tmpdir_factory, idx):
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    if idx == slice(None):
        file_adapter.export(exported_file)
    else:
        file_adapter.export(exported_file, index=idx)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]

        # only corners tested as there is no geom/gap correction
        for sub_idx in [
            (slice(None, 256), slice(None, 256)),
            (slice(None, 256), slice(-256, None)),
            (slice(-256, None), slice(None, 256)),
            (slice(-256, None), slice(-256, None)),
        ]:
            idx_res = (slice(None),) + sub_idx
            idx_mask = (idx,) + sub_idx
            assert np.allclose(res[idx_res], converted_image_stack_mask[idx_mask])


@pytest.mark.parametrize(
    "roi",
    [
        (0, 256, 0, 256),
        ((0, 256, 0, 256),),
        ((0, 256, 0, 256), (16, 32, 64, 128)),
    ],
)
def test_file_export_roi(file_adapter, tmpdir_factory, roi):
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, roi=roi)
    if isinstance(roi[0], int):
        roi = (roi,)

    with h5py.File(exported_file, "r") as h5f:
        for i in range(len(roi)):
            roi_idx = (slice(None), slice(roi[i][0], roi[i][1]), slice(roi[i][2], roi[i][3]))
            res = h5f[f"/data/{DETECTOR_NAME}:ROI_{i}/data"]
            assert np.allclose(res[:], converted_image_stack_mask[roi_idx])


@pytest.mark.parametrize(
    "downsample",
    [
        None,
        (4, 4),
    ],
)
def test_downsample(file_adapter, tmpdir_factory, downsample):
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, downsample=downsample)

    if downsample is None:
        downsample = (1, 1)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]
        ds = calc_downsample(downsample)
        # only corner tested as there is no geom/gap correction
        corner_y = 250 // downsample[0] + 1
        corner_x = 250 // downsample[1] + 1
        idx = (slice(None), slice(None, corner_y), slice(None, corner_x))
        assert np.allclose(res[idx], ds[idx])


@pytest.mark.parametrize(
    "downsample,factor",
    [
        ((5, 2), None),
        ((5, 2), 10),
    ],
)
def test_file_export_factor_n_downsample(file_adapter, tmpdir_factory, downsample, factor):
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")

    file_adapter.geometry = False
    file_adapter.gap_pixels = False
    file_adapter.export(exported_file, downsample=downsample, factor=factor)

    loc = f"/data/{DETECTOR_NAME}/data"

    with h5py.File(exported_file, "r") as h5f:
        ds = calc_downsample(downsample)
        if factor:
            ds = np.round(ds / factor).astype(np.int32)
        res = h5f[loc]
        if factor:
            assert np.allclose(res[:], ds, atol=1)
        else:
            assert np.allclose(res[:], ds)
