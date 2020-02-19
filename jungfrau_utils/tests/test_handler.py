import pytest
import numpy as np

from jungfrau_utils import JFDataHandler

DETECTOR_NAME = "JF01T03V01"
DATA_SHAPE = (3 * 512, 1024)
STACK_SHAPE = (5, *DATA_SHAPE)
DATA_SHAPE_WITH_GAPS = (3 * (512 + 2), 1024 + 6)
DATA_SHAPE_WITH_GEOMETRY = (1040 + 512, 0 + 1024)
DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY = (1040 + 512 + 2, 0 + 1024 + 6)

pedestal = np.ones((4, *DATA_SHAPE)).astype(np.float32)
gain = 10 * np.ones((4, *DATA_SHAPE)).astype(np.float32)
pixel_mask = np.random.randint(2, size=DATA_SHAPE, dtype=np.bool)

image_stack = np.arange(np.prod(STACK_SHAPE), dtype=np.uint16).reshape(STACK_SHAPE[::-1])
image_stack = np.ascontiguousarray(image_stack.transpose(2, 1, 0))

image_stack_mask = image_stack.copy()
image_stack_mask[:, pixel_mask] = 0
image_single_mask = image_stack_mask[0]

converted_image_stack = ((image_stack & 0b11111111111111).astype(np.float32) - 1) / 10

converted_image_stack_mask = converted_image_stack.copy()
converted_image_stack_mask[:, pixel_mask] = 0

image_single = image_stack[0]
converted_image_single = converted_image_stack[0]
converted_image_single_mask = converted_image_stack_mask[0]

MM_DATA_SHAPE = (2 * 512, 1024)
MM_DATA_SHAPE_WITH_GAPS = (2 * (512 + 2), 1024 + 6)
mm_image_single = image_single[: 2 * 512, :]
mm_expected_image_single = converted_image_single_mask[: 2 * 512, :]
mm_pixel_mask = pixel_mask[: 2 * 512, :]


@pytest.fixture(name="empty_handler", scope="function")
def _empty_handler():
    empty_handler = JFDataHandler(DETECTOR_NAME)

    yield empty_handler


@pytest.fixture(name="handler", scope="function")
def _handler(empty_handler):
    empty_handler.gain = gain
    empty_handler.pedestal = pedestal
    empty_handler.pixel_mask = pixel_mask

    prepared_handler = empty_handler

    yield prepared_handler


@pytest.fixture(name="handler_no_mask", scope="function")
def _handler_no_mask(empty_handler):
    empty_handler.gain = gain
    empty_handler.pedestal = pedestal

    prepared_handler = empty_handler

    yield prepared_handler


@pytest.mark.parametrize("detector_name", ["unknown_detector_name", 1, None, ""])
def test_handler_init_fail(detector_name):
    with pytest.raises(KeyError):
        JFDataHandler(detector_name)


def test_handler_init(empty_handler):
    assert empty_handler.detector_name == DETECTOR_NAME
    assert (
        empty_handler.get_shape_out(gap_pixels=True, geometry=True)
        == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    )

    assert empty_handler.pedestal is None
    assert empty_handler.gain is None
    assert empty_handler.pixel_mask is None


def test_handler_set_gain(empty_handler):
    empty_handler.gain = gain

    assert np.array_equal(empty_handler.gain, gain)
    assert empty_handler.gain.dtype == np.float32
    assert empty_handler.gain.ndim == 3
    assert empty_handler.gain.shape == (4, *DATA_SHAPE)

    empty_handler.gain = None
    assert empty_handler.gain is None

    assert empty_handler.pedestal is None
    assert empty_handler.pixel_mask is None


def test_handler_set_gain_fail(empty_handler):
    # bad gain shape
    gain_bad_shape = 100 * np.ones([1, *DATA_SHAPE]).astype(np.float32) + 1

    with pytest.raises(ValueError):
        empty_handler.gain = gain_bad_shape


def test_handler_set_pedestal(empty_handler):
    empty_handler.pedestal = pedestal

    assert np.array_equal(empty_handler.pedestal, pedestal)
    assert empty_handler.pedestal.dtype == np.float32
    assert empty_handler.pedestal.ndim == 3
    assert empty_handler.pedestal.shape == (4, *DATA_SHAPE)

    empty_handler.pedestal = None
    assert empty_handler.pedestal is None

    assert empty_handler.gain is None
    assert empty_handler.pixel_mask is None


def test_handler_set_pedestal_fail(empty_handler):
    # bad pedestal shape
    pedestal_bad_shape = 60000 * np.ones([1, *DATA_SHAPE]).astype(np.float32)

    with pytest.raises(ValueError):
        empty_handler.pedestal = pedestal_bad_shape


def test_handler_set_pixel_mask(empty_handler):
    empty_handler.pixel_mask = pixel_mask

    assert np.array_equal(empty_handler.pixel_mask, pixel_mask)
    assert empty_handler.pixel_mask.dtype == np.bool
    assert empty_handler.pixel_mask.ndim == 2
    assert empty_handler.pixel_mask.shape == DATA_SHAPE

    empty_handler.pixel_mask = None
    assert empty_handler.pixel_mask is None

    assert empty_handler.gain is None
    assert empty_handler.pedestal is None


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
    empty_handler.pedestal = pedestal
    assert empty_handler.can_convert() is False
    empty_handler.gain = gain
    assert empty_handler.can_convert() is True
    empty_handler.pedestal = None
    assert empty_handler.can_convert() is False
    empty_handler.pedestal = pedestal
    assert empty_handler.can_convert() is True
    empty_handler.gain = None
    assert empty_handler.can_convert() is False
    empty_handler.gain = gain
    assert empty_handler.can_convert() is True


def test_handler_process_single_image(handler):
    res = handler.process(image_single)

    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])


def test_handler_process_single_image_no_gaps(handler):
    res = handler.process(image_single, gap_pixels=False)

    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE_WITH_GEOMETRY

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])


def test_handler_process_single_image_no_geom(handler):
    res = handler.process(image_single, geometry=False)

    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE_WITH_GAPS

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])


def test_handler_process_single_image_no_mask(handler):
    res = handler.process(image_single, mask=False)

    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], converted_image_single[:256, :256])
    assert np.allclose(res[:256, -256:], converted_image_single[:256, -256:])
    assert np.allclose(res[-256:, :256], converted_image_single[-256:, :256])
    assert np.allclose(res[-256:, -256:], converted_image_single[-256:, -256:])


def test_handler_process_no_gaps_no_geom(handler):
    res = handler.process(image_single, gap_pixels=False, geometry=False)

    assert res.dtype == np.float32
    assert res.ndim == 2
    assert res.shape == DATA_SHAPE
    assert np.allclose(res, converted_image_single_mask)


def test_handler_process_empty_image_stack(handler):
    stack = image_stack[:0]
    res = handler.process(stack)

    assert res.dtype == np.float32
    assert res.ndim == 3
    assert res.shape == (0, *DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY)


@pytest.mark.parametrize("stack_size", [1, 5])
def test_handler_process_image_stack(handler, stack_size):
    stack = image_stack[:stack_size]
    res = handler.process(stack)

    assert res.dtype == np.float32
    assert res.ndim == 3
    assert res.shape == (stack_size, *DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY)

    # check data for submodules in all 4 corners
    assert np.allclose(res[:, :256, :256], converted_image_stack_mask[:stack_size, :256, :256])
    assert np.allclose(res[:, :256, -256:], converted_image_stack_mask[:stack_size, :256, -256:])
    assert np.allclose(res[:, -256:, :256], converted_image_stack_mask[:stack_size, -256:, :256])
    assert np.allclose(res[:, -256:, -256:], converted_image_stack_mask[:stack_size, -256:, -256:])


@pytest.mark.parametrize("conversion", [True, False])
@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_handler_process(handler, conversion, gap_pixels, geometry):
    res = handler.process(
        image_single, conversion=conversion, gap_pixels=gap_pixels, geometry=geometry
    )

    assert res.ndim == 2
    if conversion:
        assert res.dtype == np.float32
    else:
        assert res.dtype == np.uint16

    if gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    elif not gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GEOMETRY
    elif gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS
    elif not gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE


@pytest.mark.parametrize("conversion", [True, False])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
@pytest.mark.parametrize("module_map", [None, np.array([0, 1, 2])])
def test_handler_process_mm_all(handler, conversion, mask, gap_pixels, geometry, module_map):
    handler.module_map = module_map

    res = handler.process(
        image_single, conversion=conversion, mask=mask, gap_pixels=gap_pixels, geometry=geometry
    )

    assert res.ndim == 2
    if conversion:
        assert res.dtype == np.float32
    else:
        assert res.dtype == np.uint16

    if gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    elif not gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GEOMETRY
    elif gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS
    elif not gap_pixels and not geometry:
        assert res.shape == DATA_SHAPE

    # check data for submodules in all 4 corners
    if conversion:
        if mask:
            assert np.allclose(res[:256, :256], converted_image_single_mask[:256, :256])
            assert np.allclose(res[:256, -256:], converted_image_single_mask[:256, -256:])
            assert np.allclose(res[-256:, :256], converted_image_single_mask[-256:, :256])
            assert np.allclose(res[-256:, -256:], converted_image_single_mask[-256:, -256:])
        else:
            assert np.allclose(res[:256, :256], converted_image_single[:256, :256])
            assert np.allclose(res[:256, -256:], converted_image_single[:256, -256:])
            assert np.allclose(res[-256:, :256], converted_image_single[-256:, :256])
            assert np.allclose(res[-256:, -256:], converted_image_single[-256:, -256:])
    else:
        if mask:
            assert np.allclose(res[:256, :256], image_single_mask[:256, :256])
            assert np.allclose(res[:256, -256:], image_single_mask[:256, -256:])
            assert np.allclose(res[-256:, :256], image_single_mask[-256:, :256])
            assert np.allclose(res[-256:, -256:], image_single_mask[-256:, -256:])
        else:
            assert np.allclose(res[:256, :256], image_single[:256, :256])
            assert np.allclose(res[:256, -256:], image_single[:256, -256:])
            assert np.allclose(res[-256:, :256], image_single[-256:, :256])
            assert np.allclose(res[-256:, -256:], image_single[-256:, -256:])


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_handler_process_mm_missing(handler, gap_pixels, geometry):
    handler.module_map = np.array([0, 1, -1])

    res = handler.process(mm_image_single, gap_pixels=gap_pixels, geometry=geometry)

    assert res.ndim == 2
    assert res.dtype == np.float32

    if gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    elif not gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GEOMETRY
    elif gap_pixels and not geometry:
        assert res.shape == MM_DATA_SHAPE_WITH_GAPS
    elif not gap_pixels and not geometry:
        assert res.shape == MM_DATA_SHAPE

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], mm_expected_image_single[:256, :256])
    assert np.allclose(res[:256, -256:], mm_expected_image_single[:256, -256:])

    if geometry:
        assert np.allclose(res[-256:, :256], 0)
        assert np.allclose(res[-256:, -256:], 0)
    else:
        assert np.allclose(res[-256:, :256], mm_expected_image_single[-256:, :256])
        assert np.allclose(res[-256:, -256:], mm_expected_image_single[-256:, -256:])


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
@pytest.mark.parametrize("module_map", [None, np.array([0, 1, 2])])
def test_handler_shaped_pixel_mask(handler, gap_pixels, geometry, module_map):
    handler.module_map = module_map

    res = handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)

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

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], handler.pixel_mask[:256, :256])
    assert np.allclose(res[:256, -256:], handler.pixel_mask[:256, -256:])
    assert np.allclose(res[-256:, :256], handler.pixel_mask[-256:, :256])
    assert np.allclose(res[-256:, -256:], handler.pixel_mask[-256:, -256:])


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_handler_shaped_pixel_mask_mm_missing(handler, gap_pixels, geometry):
    handler.module_map = np.array([0, 1, -1])

    res = handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)

    assert res.ndim == 2
    assert res.dtype == np.bool

    if gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GAPS_WITH_GEOMETRY
    elif not gap_pixels and geometry:
        assert res.shape == DATA_SHAPE_WITH_GEOMETRY
    elif gap_pixels and not geometry:
        assert res.shape == MM_DATA_SHAPE_WITH_GAPS
    elif not gap_pixels and not geometry:
        assert res.shape == MM_DATA_SHAPE

    # check data for submodules in all 4 corners
    assert np.allclose(res[:256, :256], mm_pixel_mask[:256, :256])
    assert np.allclose(res[:256, -256:], mm_pixel_mask[:256, -256:])

    if geometry:
        assert np.allclose(res[-256:, :256], True)
        assert np.allclose(res[-256:, -256:], True)
    else:
        assert np.allclose(res[-256:, :256], mm_pixel_mask[-256:, :256])
        assert np.allclose(res[-256:, -256:], mm_pixel_mask[-256:, -256:])


def test_handler_get_gains(handler):
    res = handler.get_gains(image_stack, mask=False, gap_pixels=False, geometry=False)

    assert (res >= 0).all() and (res <= 3).all()


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.uint32, np.uint64, np.int16, np.int32, np.int64],
)
def test_handler_get_gains_fail(handler, dtype):
    bad_data = image_stack.astype(dtype)
    with pytest.raises(TypeError):
        handler.get_gains(bad_data, mask=False, gap_pixels=False, geometry=False)


def test_handler_get_saturated_value(handler):
    assert handler.get_saturated_value() == 49152


def test_handler_get_saturated_value_highgain(handler):
    handler.highgain = True
    assert handler.get_saturated_value() == 16383
