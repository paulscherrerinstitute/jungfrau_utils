import h5py
import numpy as np
import pytest

from jungfrau_utils import File, JFDataHandler


@pytest.fixture(name="gain_file", scope="module")
def _gain_file(tmpdir_factory, request):
    gain_file = tmpdir_factory.mktemp("data").join("gains.h5")

    DATA_SHAPE = getattr(request.module, "DATA_SHAPE")
    with h5py.File(gain_file, "w") as h5f:
        h5f["/gains"] = getattr(request.module, "gain")

    return gain_file


@pytest.fixture(name="pedestal_file", scope="module")
def _pedestal_file(tmpdir_factory, request):
    pedestal_file = tmpdir_factory.mktemp("data").join("pedestal.h5")

    DATA_SHAPE = getattr(request.module, "DATA_SHAPE")
    with h5py.File(pedestal_file, "w") as h5f:
        h5f["/gains"] = np.ones((4, *DATA_SHAPE)).astype(np.float32)
        h5f["/pixel_mask"] = getattr(request.module, "pixel_mask_orig")

    return pedestal_file


@pytest.fixture(name="jungfrau_file", scope="module")
def _jungfrau_file(tmpdir_factory, request):
    jungfrau_file = tmpdir_factory.mktemp("data").join("test_jf.h5")

    DETECTOR_NAME = getattr(request.module, "DETECTOR_NAME")
    STACK_SHAPE = getattr(request.module, "STACK_SHAPE")
    with h5py.File(jungfrau_file, "w") as h5f:
        h5f["/general/detector_name"] = bytes(DETECTOR_NAME, encoding="utf-8")
        h5f[f"/data/{DETECTOR_NAME}/daq_rec"] = 3840 * np.ones((STACK_SHAPE[0], 1)).astype(np.int64)
        h5f[f"/data/{DETECTOR_NAME}/data"] = getattr(request.module, "image_stack")
        h5f[f"/data/{DETECTOR_NAME}/is_good_frame"] = np.ones((STACK_SHAPE[0], 1)).astype(bool)

    return jungfrau_file


@pytest.fixture(name="file_adapter", scope="module")
def _file_adapter(jungfrau_file, gain_file, pedestal_file):
    file_adapter = File(jungfrau_file, gain_file=gain_file, pedestal_file=pedestal_file)

    yield file_adapter


@pytest.fixture(name="empty_handler", scope="function")
def _empty_handler(request):

    DETECTOR_NAME = getattr(request.module, "DETECTOR_NAME")
    empty_handler = JFDataHandler(DETECTOR_NAME)

    yield empty_handler


@pytest.fixture(name="handler", scope="function")
def _handler(empty_handler, request):

    empty_handler.gain = getattr(request.module, "gain")
    empty_handler.pedestal = getattr(request.module, "pedestal")
    empty_handler.pixel_mask = getattr(request.module, "pixel_mask_orig")

    prepared_handler = empty_handler

    yield prepared_handler


@pytest.fixture(name="handler_no_mask", scope="function")
def _handler_no_mask(empty_handler, request):
    empty_handler.gain = getattr(request.module, "gain")
    empty_handler.pedestal = getattr(request.module, "pedestal")

    prepared_handler = empty_handler

    yield prepared_handler
