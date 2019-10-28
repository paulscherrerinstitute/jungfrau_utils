import pytest
import numpy as np

from jungfrau_utils import JFDataHandler

DETECTOR_NAME = 'JF01T03V01'
DATA_SHAPE = (3 * 512, 1024)
DATA_SHAPE_WITH_GAPS = (3 * (512 + 2), 1024 + 6)
DATA_SHAPE_WITH_GEOMETRY = (1040 + 512, 0 + 1024)
DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY = (1040 + 512 + 2, 0 + 1024 + 6)

pedestal = 60000 * np.random.random(size=[4, *DATA_SHAPE]).astype(np.float32)
gain = 100 * np.random.random(size=[4, *DATA_SHAPE]).astype(np.float32) + 1
pixel_mask = np.random.randint(2, size=DATA_SHAPE, dtype=np.bool)
data = np.zeros(DATA_SHAPE, dtype=np.uint16)


@pytest.fixture(scope='function')
def empty_handler():
    empty_handler = JFDataHandler(DETECTOR_NAME)

    yield empty_handler


@pytest.fixture(scope='function')
def handler(empty_handler):
    empty_handler.G = gain
    empty_handler.P = pedestal
    empty_handler.pixel_mask = pixel_mask

    prepared_handler = empty_handler

    yield prepared_handler


@pytest.fixture(scope='function')
def handler_no_mask(empty_handler):
    empty_handler.G = gain
    empty_handler.P = pedestal

    prepared_handler = empty_handler

    yield prepared_handler


@pytest.mark.parametrize("detector_name", ["unknown_detector_name", 1, None, ""])
def test_handler_init_fail(detector_name):
    with pytest.raises(KeyError):
        JFDataHandler(detector_name)


def test_handler_init(empty_handler):
    assert empty_handler.detector_name == DETECTOR_NAME
    assert empty_handler.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY

    assert empty_handler.P is None
    assert empty_handler.G is None
    assert empty_handler.pixel_mask is None


def test_handler_set_gain(empty_handler):
    empty_handler.G = gain

    assert np.array_equal(empty_handler.G, gain)
    assert empty_handler.G.dtype == np.float32
    assert empty_handler.G.ndim == 3
    assert empty_handler.G.shape == (4, *DATA_SHAPE)

    empty_handler.G = None
    assert empty_handler.G is None

    assert empty_handler.P is None
    assert empty_handler.pixel_mask is None


def test_handler_set_gain_fail(empty_handler):
    # bad gain shape
    gain = 100 * np.ones([1, *DATA_SHAPE]).astype(np.float32) + 1

    with pytest.raises(ValueError):
        empty_handler.G = gain


def test_handler_set_pedestal(empty_handler):
    empty_handler.P = pedestal

    assert np.array_equal(empty_handler.P, pedestal)
    assert empty_handler.P.dtype == np.float32
    assert empty_handler.P.ndim == 3
    assert empty_handler.P.shape == (4, *DATA_SHAPE)

    empty_handler.P = None
    assert empty_handler.P is None

    assert empty_handler.G is None
    assert empty_handler.pixel_mask is None


def test_handler_set_pedestal_fail(empty_handler):
    # bad pedestal shape
    pedestal = 60000 * np.ones([1, *DATA_SHAPE]).astype(np.float32)

    with pytest.raises(ValueError):
        empty_handler.P = pedestal


def test_handler_set_pixel_mask(empty_handler):
    pixel_mask = np.random.randint(2, size=DATA_SHAPE, dtype=np.bool)
    empty_handler.pixel_mask = pixel_mask

    assert np.array_equal(empty_handler.pixel_mask, pixel_mask)
    assert empty_handler.pixel_mask.dtype == np.bool
    assert empty_handler.pixel_mask.ndim == 2
    assert empty_handler.pixel_mask.shape == DATA_SHAPE

    empty_handler.pixel_mask = None
    assert empty_handler.pixel_mask is None

    assert empty_handler.G is None
    assert empty_handler.P is None


def test_handler_set_highgain(empty_handler):
    empty_handler.highgain = True
    assert empty_handler.highgain is True
    empty_handler.highgain = False
    assert empty_handler.highgain is False


@pytest.mark.parametrize("mm_value", [np.array([-1, 0, 1]), np.array([0, 1, 2]), None])
def test_handler_set_module_map(empty_handler, mm_value):
    empty_handler.module_map = mm_value
    if mm_value is None:
        # setting module_map to None should emulate 'all modules are present'
        assert np.array_equal(empty_handler.module_map, np.array([0, 1, 2]))
    else:
        assert np.array_equal(empty_handler.module_map, mm_value)


@pytest.mark.parametrize("mm_value", [np.array(["0", "1", "2"])])
def test_handler_set_module_map_type_error(empty_handler, mm_value):
    with pytest.raises(TypeError):
        empty_handler.module_map = mm_value


@pytest.mark.parametrize("mm_value", [np.array([-2, 0, 1]), np.array([0, 1, 1000])])
def test_handler_set_module_map_value_error(empty_handler, mm_value):
    with pytest.raises(ValueError):
        empty_handler.module_map = mm_value


def test_handler_can_convert(empty_handler):
    assert empty_handler.can_convert() is False
    empty_handler.P = pedestal
    assert empty_handler.can_convert() is False
    empty_handler.G = gain
    assert empty_handler.can_convert() is True
    empty_handler.P = None
    assert empty_handler.can_convert() is False
    empty_handler.P = pedestal
    assert empty_handler.can_convert() is True
    empty_handler.G = None
    assert empty_handler.can_convert() is False
    empty_handler.G = gain
    assert empty_handler.can_convert() is True


def test_handler_process_single_image(handler):
    data = np.zeros(DATA_SHAPE, dtype=np.uint16)
    res = handler.process(data)

    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY


@pytest.mark.parametrize("stack_size", [0, 1, 10])
def test_handler_process_image_stack(handler, stack_size):
    data = np.zeros([stack_size, *DATA_SHAPE], dtype=np.uint16)
    res = handler.process(data)

    assert res.dtype == np.float32
    assert res.ndim == 3
    assert res.shape == (stack_size, *DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY)


@pytest.mark.parametrize("convertion", [True, False])
@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_handler_process(handler, convertion, gap_pixels, geometry):
    handler.convertion = convertion
    handler.gap_pixels = gap_pixels
    handler.geometry = geometry

    res = handler.process(data)

    assert res.ndim == 2
    if convertion:
        assert res.dtype == np.dtype(np.float32)
    else:
        assert res.dtype == np.dtype(np.uint16)

    if gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    elif not gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GEOMETRY
    elif gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS
    elif not gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE


def test_handler_process_no_gaps_no_geom(handler):
    handler.gap_pixels = False
    handler.geometry = False

    res = handler.process(data)
    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
@pytest.mark.parametrize("module_mask", [None, np.array([0, -1, 1])])
def test_handler_shaped_pixel_mask(handler, gap_pixels, geometry, module_mask):
    handler.gap_pixels = gap_pixels
    handler.geometry = geometry
    handler.module_map = module_mask

    res = handler.shaped_pixel_mask

    assert res.ndim == 2
    assert res.dtype == np.bool

    if gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    elif not gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GEOMETRY
    elif gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS
    elif not gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE
