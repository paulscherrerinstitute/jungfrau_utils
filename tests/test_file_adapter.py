import h5py
import numpy as np
import pytest

from jungfrau_utils import File

DETECTOR_NAME = "JF01T03V01"
DATA_SHAPE = (3 * 512, 1024)
STACK_SHAPE = (10, *DATA_SHAPE)

DATA_SHAPE_WITH_GAPS = (3 * (512 + 2), 1024 + 6)
DATA_SHAPE_WITH_GEOMETRY = (1100 + 512, 0 + 1024)
DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY = (1100 + 512 + 2, 0 + 1024 + 6)

IMAGE_SHAPE = DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
STACK_IMAGE_SHAPE = (3, *DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY)

pixel_mask_orig = np.random.randint(2, size=DATA_SHAPE, dtype=np.uint32)
pixel_mask = pixel_mask_orig.astype(bool, copy=True)

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


def test_file_adapter(file_adapter, gain_file, pedestal_file):
    assert file_adapter.gain_file == gain_file
    assert file_adapter.pedestal_file == pedestal_file


def test_file_get_index_image(file_adapter):
    res = file_adapter[0]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 2
    assert res.shape == IMAGE_SHAPE

    assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])

    res = file_adapter[0, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 2
    assert res.shape == IMAGE_SHAPE

    assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])

    res = file_adapter[0, :, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 2
    assert res.shape == IMAGE_SHAPE

    assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])


def test_file_get_slice_image(file_adapter):
    res = file_adapter[:3]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[:3, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[:3, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[:3, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[:3, -256:, -256:])

    res = file_adapter[:3, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[:3, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[:3, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[:3, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[:3, -256:, -256:])

    res = file_adapter[:3, :, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[:3, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[:3, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[:3, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[:3, -256:, -256:])


def test_file_get_fancy_index_list_image(file_adapter):
    indices = [0, 2, 4]

    res = file_adapter[indices]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])

    res = file_adapter[indices, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])

    res = file_adapter[indices, :, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])


def test_file_get_fancy_index_tuple_image(file_adapter):
    indices = (0, 2, 4)

    # this is a special case, but has the same behaviour as h5py
    res = file_adapter[indices]
    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 0
    assert res.shape == ()

    res = file_adapter[indices, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])

    res = file_adapter[indices, :, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])


def test_file_get_fancy_index_range_image(file_adapter):
    indices = range(0, 5, 2)
    res = file_adapter[indices]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])

    res = file_adapter[indices, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])

    res = file_adapter[indices, :, :]

    assert res.dtype == np.dtype(np.float32)
    assert res.ndim == 3
    assert res.shape == STACK_IMAGE_SHAPE

    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])


def test_file_export(file_adapter, tmpdir_factory):
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]
        assert np.allclose(res[:, :256, :256], converted_image_stack_mask[:, :256, :256])
        assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[:, :256, -256:])
        assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[:, -256:, :256])
        assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[:, -256:, -256:])


def test_file_export_index1(file_adapter, tmpdir_factory):
    indices = [0, 2, 4]
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, index=indices)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]
        assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
        assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
        assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
        assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])


def test_file_export_index2(file_adapter, tmpdir_factory):
    indices = (0, 2, 4)
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, index=indices)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]
        assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
        assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
        assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
        assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])


def test_file_export_index3(file_adapter, tmpdir_factory):
    indices = range(0, 5, 2)
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, index=indices)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}/data"]
        assert np.allclose(res[:, :256, :256], converted_image_stack_mask[indices, :256, :256])
        assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[indices, :256, -256:])
        assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[indices, -256:, :256])
        assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[indices, -256:, -256:])


def test_file_export_roi1(file_adapter, tmpdir_factory):
    roi = (0, 256, 0, 256)
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, roi=roi)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}:ROI_0/data"]
        assert np.allclose(res[:], converted_image_stack_mask[:, :256, :256])


def test_file_export_roi2(file_adapter, tmpdir_factory):
    roi = ((0, 256, 0, 256),)
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, roi=roi)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}:ROI_0/data"]
        assert np.allclose(res[:], converted_image_stack_mask[:, :256, :256])


def test_file_export_roi3(file_adapter, tmpdir_factory):
    roi = ((0, 256, 0, 256), (16, 32, 64, 128))
    exported_file = tmpdir_factory.mktemp("export").join("test.h5")
    file_adapter.export(exported_file, roi=roi)

    with h5py.File(exported_file, "r") as h5f:
        res = h5f[f"/data/{DETECTOR_NAME}:ROI_0/data"]
        assert np.allclose(res[:], converted_image_stack_mask[:, :256, :256])

        res = h5f[f"/data/{DETECTOR_NAME}:ROI_1/data"]
        assert np.allclose(res[:], converted_image_stack_mask[:, 16:32, 64:128])
