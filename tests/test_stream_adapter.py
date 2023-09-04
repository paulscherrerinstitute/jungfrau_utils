import h5py
import numpy as np
import pytest

from jungfrau_utils import JFDataHandler, StreamAdapter

DETECTOR_NAME = "JF01T03V01"
DATA_SHAPE = (3 * 512, 1024)
DATA_SHAPE_WITH_GAPS = (3 * (512 + 2), 1024 + 6)
DATA_SHAPE_WITH_GEOMETRY = (1100 + 512, 0 + 1024)
DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY = (1100 + 512 + 2, 0 + 1024 + 6)

image = np.arange(np.prod(DATA_SHAPE), dtype=np.uint16).reshape(DATA_SHAPE[::-1])
image = np.ascontiguousarray(image.transpose())


@pytest.fixture(name="gain_file", scope="module")
def _gain_file(tmpdir_factory):
    gain_filename = tmpdir_factory.mktemp("data").join("gains.h5")

    with h5py.File(gain_filename, "w") as h5f:
        h5f["/gains"] = 10 * np.ones((4, *DATA_SHAPE))

    return gain_filename


@pytest.fixture(name="bad_gain_file", scope="module")
def _bad_gain_file(tmpdir_factory):
    gain_filename = tmpdir_factory.mktemp("data").join("gains.h5")

    with h5py.File(gain_filename, "w") as h5f:
        h5f["/gains"] = 10 * np.ones((4, 2 * 512, 1024))

    return gain_filename


@pytest.fixture(name="pedestal_file", scope="module")
def _pedestal_file(tmpdir_factory):
    pedestal_filename = tmpdir_factory.mktemp("data").join("gains.h5")

    with h5py.File(pedestal_filename, "w") as h5f:
        h5f["/gains"] = np.ones((4, *DATA_SHAPE))
        h5f["/pixel_mask"] = np.random.randint(2, size=DATA_SHAPE, dtype=bool)

    return pedestal_filename


@pytest.fixture(name="bad_pedestal_file", scope="module")
def _bad_pedestal_file(tmpdir_factory):
    pedestal_filename = tmpdir_factory.mktemp("data").join("gains.h5")

    with h5py.File(pedestal_filename, "w") as h5f:
        h5f["/gains"] = np.ones((4, 2 * 512, 1024))
        h5f["/pixel_mask"] = np.random.randint(2, size=DATA_SHAPE, dtype=bool)

    return pedestal_filename


@pytest.fixture(name="stream_adapter", scope="function")
def _stream_handler():
    stream_adapter = StreamAdapter()

    yield stream_adapter


@pytest.mark.parametrize("detector_name", ["JF01T03V01"])
def test_stream_set_detector_name(stream_adapter, detector_name):
    metadata = {"detector_name": detector_name}

    stream_adapter.process(image, metadata)

    assert isinstance(stream_adapter.handler, JFDataHandler)


@pytest.mark.parametrize("detector_name", ["", None])
def test_stream_set_detector_name_empty(stream_adapter, detector_name):
    metadata = {"detector_name": detector_name}

    stream_adapter.process(image, metadata)

    assert stream_adapter.handler is None


@pytest.mark.parametrize("detector_name", ["unknown_detector_name", 1])
def test_stream_set_detector_name_fail(stream_adapter, caplog, detector_name):
    metadata = {"detector_name": detector_name}

    res = stream_adapter.process(image, metadata)

    assert "Error creating data handler" in caplog.text
    assert res.dtype == np.dtype(np.uint16)


def test_stream_gain_file_fail(stream_adapter, caplog, bad_gain_file, pedestal_file):
    metadata = {
        "detector_name": DETECTOR_NAME,
        "gain_file": bad_gain_file,
        "pedestal_file": pedestal_file,
    }

    res = stream_adapter.process(image, metadata)

    assert "Error loading gain file" in caplog.text
    assert res.dtype == np.dtype(np.uint16)


def test_stream_pedestal_file_fail(stream_adapter, caplog, gain_file, bad_pedestal_file):
    metadata = {
        "detector_name": DETECTOR_NAME,
        "gain_file": gain_file,
        "pedestal_file": bad_pedestal_file,
    }

    res = stream_adapter.process(image, metadata)

    assert "Error loading pedestal file" in caplog.text
    assert res.dtype == np.dtype(np.uint16)


def test_stream_process(stream_adapter, gain_file, pedestal_file):
    metadata = {
        "detector_name": DETECTOR_NAME,
        "gain_file": gain_file,
        "pedestal_file": pedestal_file,
    }

    res = stream_adapter.process(image, metadata)

    assert res.dtype == np.dtype(np.float32)
    assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY


@pytest.mark.parametrize("daq_rec", [0, 1])
def test_stream_highgain(stream_adapter, gain_file, pedestal_file, daq_rec):
    metadata = {
        "detector_name": DETECTOR_NAME,
        "gain_file": gain_file,
        "pedestal_file": pedestal_file,
        "daq_rec": daq_rec,
    }

    res = stream_adapter.process(image, metadata)

    assert stream_adapter.handler.highgain == bool(daq_rec & 0b1)
    assert res.dtype == np.dtype(np.float32)
    assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY


def test_stream_module_map(stream_adapter, gain_file, pedestal_file):
    metadata = {
        "detector_name": DETECTOR_NAME,
        "gain_file": gain_file,
        "pedestal_file": pedestal_file,
        "module_map": np.array([-1, 0, 1]),
    }

    data = np.zeros((2 * 512, 1024), np.uint16)
    res = stream_adapter.process(data, metadata)

    assert res.dtype == np.dtype(np.float32)
    assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
