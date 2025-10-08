import numpy as np

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

rng = np.random.default_rng()
pixel_mask_orig = rng.integers(2, size=DATA_SHAPE, dtype=np.uint32)
pixel_mask = pixel_mask_orig.astype(bool, copy=True)
inv_pixel_mask = np.invert(pixel_mask)

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

stack_index = [0, 3, 5]

pedestal = np.ones((4, *DATA_SHAPE)).astype(np.float32)
gain = 10 * np.ones((4, *DATA_SHAPE)).astype(np.float32)

