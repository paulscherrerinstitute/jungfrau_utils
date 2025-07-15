import numpy as np
import pytest

from jungfrau_utils import JFDataHandler
from jungfrau_utils.helpers import reverse_pixel_mask

DETECTOR_NAME = "JF01T03V01"
DATA_SHAPE = (3 * 512, 1024)

pixel_mask_orig = np.random.randint(2, size=DATA_SHAPE, dtype=np.uint32)
pixel_mask = pixel_mask_orig.astype(bool, copy=True)

STRIPSEL_DETECTOR_NAME = "JF12T04V01"
STRIPSEL_DATA_SHAPE = (4 * 512, 1024)

# TODO: a better test, as only the top half of the stripsel module is used
stripsel_pixel_mask_orig = np.ones(STRIPSEL_DATA_SHAPE, dtype=np.uint32)
stripsel_pixel_mask = stripsel_pixel_mask_orig.astype(bool, copy=True)


@pytest.fixture(name="handler", scope="function")
def _handler():
    handler = JFDataHandler(DETECTOR_NAME)
    handler.pixel_mask = pixel_mask_orig

    yield handler


@pytest.fixture(name="stripsel_handler", scope="function")
def _stripsel_handler():
    handler = JFDataHandler(STRIPSEL_DETECTOR_NAME)
    handler.pixel_mask = stripsel_pixel_mask_orig

    yield handler


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_reverse_pixel_mask(handler, gap_pixels, geometry):
    processed_mask = handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)
    reversed_pixel_mask = reverse_pixel_mask(DETECTOR_NAME, processed_mask, gap_pixels, geometry)

    assert np.array_equal(pixel_mask, reversed_pixel_mask)


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_reverse_pixel_mask2(handler, gap_pixels, geometry):
    processed_mask = handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)
    processed_mask[100:200, :] = False
    processed_mask[:, 100:200] = False
    # can't unmask the empty areas!
    processed_mask[10:60, 10:60] = True

    reversed_pixel_mask = reverse_pixel_mask(DETECTOR_NAME, processed_mask, gap_pixels, geometry)

    handler.pixel_mask = reversed_pixel_mask
    converted_mask_check = handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)

    assert np.array_equal(processed_mask, converted_mask_check)


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_reverse_pixel_mask_stripsel(stripsel_handler, gap_pixels, geometry):
    processed_mask = stripsel_handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)

    reversed_pixel_mask = reverse_pixel_mask(
        STRIPSEL_DETECTOR_NAME, processed_mask, gap_pixels, geometry
    )

    assert np.array_equal(stripsel_pixel_mask, reversed_pixel_mask)


@pytest.mark.parametrize("gap_pixels", [True, False])
@pytest.mark.parametrize("geometry", [True, False])
def test_reverse_pixel_mask_stripsel2(stripsel_handler, gap_pixels, geometry):
    processed_mask = stripsel_handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)
    processed_mask[10:30, 10:30] = False
    # can't unmask the empty areas!
    processed_mask[30:60, 30:60] = True

    reversed_pixel_mask = reverse_pixel_mask(
        STRIPSEL_DETECTOR_NAME, processed_mask, gap_pixels, geometry
    )

    stripsel_handler.pixel_mask = reversed_pixel_mask
    converted_mask_check = stripsel_handler.get_pixel_mask(gap_pixels=gap_pixels, geometry=geometry)

    assert np.array_equal(processed_mask, converted_mask_check)
