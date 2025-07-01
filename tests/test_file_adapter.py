import h5py
import numpy as np
import pytest

from jungfrau_utils import File

DETECTOR_NAME = "JF01T03V01"
DATA_SHAPE = (3 * 512, 1024)
STACK_SHAPE = (10, *DATA_SHAPE)

DATA_SHAPE_WITH_GAPS = (3 * (512 + 2), 1024 + 6)
DATA_SHAPE_WITH_GEOMETRY = (1100 + 512, 0 + 1024)  # 3rd corner pos + module size
DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY = (
    1100 + 512 + 2,
    0 + 1024 + 6,
)  # 3rd corner + module + chip gaps

IMAGE_SHAPE = DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
STACK_IMAGE_SHAPE = (3, *DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY)

pixel_mask_orig = np.random.randint(2, size=DATA_SHAPE, dtype=np.uint32)
pixel_mask = pixel_mask_orig.astype(bool, copy=True)
inv_pixel_mask = np.invert(pixel_mask)

image_stack = np.arange(np.prod(STACK_SHAPE), dtype=np.uint16).reshape(STACK_SHAPE[::-1])
image_stack = np.ascontiguousarray(image_stack.transpose(2, 1, 0))

converted_image_stack = ((image_stack & 0b11111111111111).astype(np.float32) - 1) / 10

converted_image_stack_mask = converted_image_stack.copy()
converted_image_stack_mask[:, pixel_mask] = 0

image_single = image_stack[0]
converted_image_single = converted_image_stack[0]
converted_image_single_mask = converted_image_stack_mask[0]

stack_index = [0, 3, 5]


@pytest.fixture(name="gain_file", scope="module")
def _gain_file(tmpdir_factory):
    gain_file = tmpdir_factory.mktemp("data").join("gains.h5")

    with h5py.File(gain_file, "w") as h5f:
        h5f["/gains"] = 10 * np.ones((4, *DATA_SHAPE)).astype(np.float32)

    return gain_file


@pytest.fixture(name="pedestal_file", scope="module")
def _pedestal_file(tmpdir_factory):
    pedestal_file = tmpdir_factory.mktemp("data").join("pedestal.h5")

    with h5py.File(pedestal_file, "w") as h5f:
        h5f["/gains"] = np.ones((4, *DATA_SHAPE)).astype(np.float32)
        h5f["/pixel_mask"] = pixel_mask_orig

    return pedestal_file


@pytest.fixture(name="jungfrau_file", scope="module")
def _jungfrau_file(tmpdir_factory):
    jungfrau_file = tmpdir_factory.mktemp("data").join("test_jf.h5")

    with h5py.File(jungfrau_file, "w") as h5f:
        h5f["/general/detector_name"] = bytes(DETECTOR_NAME, encoding="utf-8")
        h5f[f"/data/{DETECTOR_NAME}/daq_rec"] = 3840 * np.ones((STACK_SHAPE[0], 1)).astype(np.int64)
        h5f[f"/data/{DETECTOR_NAME}/data"] = image_stack
        h5f[f"/data/{DETECTOR_NAME}/is_good_frame"] = np.ones((STACK_SHAPE[0], 1)).astype(bool)

    return jungfrau_file


@pytest.fixture(name="file_adapter", scope="module")
def _file_adapter(jungfrau_file, gain_file, pedestal_file):
    file_adapter = File(jungfrau_file, gain_file=gain_file, pedestal_file=pedestal_file)

    yield file_adapter

def calc_downsample(downsample, roi):
    if roi is None:
        roi=(0, DATA_SHAPE[0], 0, DATA_SHAPE[1])

    ds_shape = (
        (roi[1] - roi[0] + downsample[0] - 1) // downsample[0],
        (roi[3] - roi[2] + downsample[1] - 1) // downsample[1],
    )
    ds = np.zeros((STACK_SHAPE[0], *ds_shape),dtype=np.float32)

    if downsample == (DATA_SHAPE[0], DATA_SHAPE[1]):
        ds_px_count = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    else:
        ds_px_count = downsample[0] * downsample[1]
    for j in range(ds_shape[0]):
        i_y = downsample[0] * j + roi[0]
        rng_y = slice(i_y, min(i_y + downsample[0], roi[1]))
        for k in range(ds_shape[1]):
            i_x = downsample[1] * k + roi[2]
            rng_x = slice(i_x, min(i_x + downsample[1], roi[3]))
            gpr = np.count_nonzero(inv_pixel_mask[rng_y, rng_x]) / ds_px_count
            ds[:, j, k] = (
                np.sum(converted_image_stack_mask[:, rng_y, rng_x], axis=(1, 2)) / gpr
                if gpr
                else 0
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
        downsample = ((1, 1),)
    if isinstance(downsample[0], int):
        downsample = (downsample,)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]
        ds=calc_downsample(downsample[0], None)
        # only corner tested as there is no geom/gap correction
        corner_y = 250 // downsample[0][0] + 1
        corner_x = 250 // downsample[0][1] + 1
        idx = (slice(None), slice(None, corner_y), slice(None, corner_x))
        assert np.allclose(res[idx], ds[idx])
