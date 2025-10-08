from __future__ import annotations

import numbers
import struct
import warnings
from itertools import islice
from typing import Iterable

import bitshuffle
import h5py
import numpy as np
from bitshuffle.h5 import H5_COMPRESS_LZ4, H5FILTER  # pylint: disable=no-name-in-module
from numba import njit, prange
from numpy.typing import NDArray

import jungfrau_utils
from jungfrau_utils.data_handler import JFDataHandler
from jungfrau_utils.swissfel_helpers import (
    get_single_detector_name,
    locate_gain_file,
    locate_pedestal_file,
)

warnings.filterwarnings("default", category=DeprecationWarning)

# bitshuffle hdf5 filter params
BLOCK_SIZE = 32768
compargs = {"compression": H5FILTER, "compression_opts": (BLOCK_SIZE, H5_COMPRESS_LZ4)}


class File:
    """Jungfrau file wrapper.

    Args:
        file_path (str): Path to Jungfrau file
        detector_name (str, optional): Name of a detector, which data should be processed if there
            are multiple detectors' data present in the file. If empty, the file must contain data
            for a single detector only. Defaults to ''.
        gain_file (str, optional): Path to gain file. Auto-locate if empty. Defaults to ''.
        pedestal_file (str, optional): Path to pedestal file. Auto-locate if empty. Defaults to ''.
        conversion (bool, optional): Apply gain conversion and pedestal correction.
            Defaults to True.
        mask (bool, optional): Perform masking of bad pixels (assign them to 0). Defaults to True.
        gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
        double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
            "keep", "mask", or "interp". Defaults to "keep".
        geometry (bool, optional): Apply geometry correction. Defaults to True.
        parallel (bool, optional): Use parallelized processing. Defaults to True.
    """

    def __init__(
        self,
        file_path: str,
        *,
        detector_name: str = "",
        gain_file: str = "",
        pedestal_file: str = "",
        conversion: bool = True,
        mask: bool = True,
        gap_pixels: bool = True,
        double_pixels: str = "keep",
        geometry: bool = True,
        parallel: bool = True,
    ):
        self.file = h5py.File(file_path, "r")

        if not detector_name:
            detector_name = get_single_detector_name(file_path)

        # placeholders for processed files
        self.handler: JFDataHandler = JFDataHandler(detector_name)
        self._detector_name: str = detector_name

        self._conversion = conversion
        self._mask = mask
        self._gap_pixels = gap_pixels
        self._double_pixels = double_pixels
        self._geometry = geometry
        self._parallel = parallel

        # No need for any further setup if the file is already processed
        if self._processed:
            if jungfrau_utils.verbose:
                print("The file is already converted, setting processing parameters has no effect.")
                print("No gain/pedestal files are loaded.")
            return

        # Gain file
        if not gain_file:
            gain_file = locate_gain_file(
                file_path, detector_name=detector_name, verbose=jungfrau_utils.verbose
            )

        self.handler.gain_file = gain_file

        # Pedestal file (with a pixel mask)
        if not pedestal_file:
            pedestal_file = locate_pedestal_file(
                file_path, detector_name=detector_name, verbose=jungfrau_utils.verbose
            )

        self.handler.pedestal_file = pedestal_file

        if "module_map" in self._meta_group:
            module_map = self._meta_group["module_map"][:]
            if module_map.ndim == 2:
                # This is an old format. Pick only the first row (module_map of the first frame),
                # because it is not expected that module_map ever changes during a run.
                module_map = module_map[0, :]
        else:
            module_map = None

        self.handler.module_map = module_map

        # TODO: Here we use daq_rec only of the first pulse, where is_good_frame is True, within
        # an hdf5 file, however its value can be different for later pulses and this needs to be
        # taken care of.
        good_frame_idx = np.nonzero(self._data_group["is_good_frame"])[0]
        if good_frame_idx.size > 0:
            daq_rec = self._data_group["daq_rec"][good_frame_idx[0]]
        else:
            warnings.warn("The file doesn't contain good frames. Highgain is set to False.")
            daq_rec = 0

        self.handler.highgain = daq_rec & 0b1

    @property
    def file_path(self) -> str:
        warnings.warn(
            "file_path is deprecated and will be removed in jungfrau_utils/4.0", DeprecationWarning
        )
        return self.file.filename

    @property
    def detector_name(self) -> str:
        """Detector name (readonly)."""
        return self._detector_name

    @property
    def gain_file(self) -> str:
        """Gain file path (readonly)."""
        return self.handler.gain_file

    @property
    def pedestal_file(self) -> str:
        """Pedestal file path (readonly)."""
        return self.handler.pedestal_file

    @property
    def conversion(self) -> bool:
        """A flag for applying pedestal correction and gain conversion."""
        return self._conversion

    @conversion.setter
    def conversion(self, value: bool) -> None:
        if self._processed:
            print("The file is already processed, setting 'conversion' has no effect.")
            return

        self._conversion = value

    @property
    def mask(self) -> bool:
        """A flag for masking bad pixels."""
        return self._mask

    @mask.setter
    def mask(self, value: bool) -> None:
        if self._processed:
            print("The file is already processed, setting 'mask' has no effect.")
            return

        self._mask = value

    @property
    def gap_pixels(self) -> bool:
        """A flag for adding gap pixels."""
        return self._gap_pixels

    @gap_pixels.setter
    def gap_pixels(self, value: bool) -> None:
        if self._processed:
            print("The file is already processed, setting 'gap_pixels' has no effect.")
            return

        self._gap_pixels = value

    @property
    def double_pixels(self) -> str:
        """A parameter for making modifications to double pixels."""
        return self._double_pixels

    @double_pixels.setter
    def double_pixels(self, value: str) -> None:
        if self._processed:
            print("The file is already processed, setting 'double_pixels' has no effect.")
            return

        self._double_pixels = value

    @property
    def geometry(self) -> bool:
        """A flag for applying geometry."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: bool) -> None:
        if self._processed:
            print("The file is already processed, setting 'geometry' has no effect.")
            return

        self._geometry = value

    @property
    def parallel(self) -> bool:
        """A flag for using parallelization."""
        return self._parallel

    @parallel.setter
    def parallel(self, value: bool) -> None:
        if self._processed:
            print("The file is already processed, setting 'parallel' has no effect.")
            return

        self._parallel = value

    @property
    def _processed(self) -> bool:
        return "conversion_factor" in self._meta_group

    @property
    def _data_dset(self) -> h5py.Dataset:
        return self.file[f"data/{self.detector_name}/data"]

    @property
    def _data_group(self) -> h5py.Group:
        return self.file[f"data/{self.detector_name}"]

    @property
    def _meta_group(self) -> h5py.Group:
        if "meta" in self._data_group and isinstance(self._data_group["meta"], h5py.Group):
            return self._data_group["meta"]
        return self._data_group

    def get_shape_out(self) -> tuple[int, int]:
        """Return the final image shape of a detector, based on gap_pixel and geometry flags.

        Returns:
            tuple: Height and width of a resulting image.
        """
        if self._processed:
            return self._data_dset.shape[-2:]

        return self.handler.get_shape_out(gap_pixels=self.gap_pixels, geometry=self.geometry)

    def get_dtype_out(self) -> np.dtype:
        """Return resulting image dtype of a detector.

        Returns:
            dtype: dtype of a resulting image.
        """
        if self._processed:
            return self._data_dset.dtype

        return self.handler.get_dtype_out(self._data_dset.dtype, conversion=self.conversion)

    def get_pixel_mask(self) -> NDArray | None:
        """Return pixel mask, shaped according to gap_pixel and geometry flags.

        Returns:
            ndarray: Resulting pixel mask, where True values correspond to valid pixels.
        """
        if self._processed:
            return self._meta_group["pixel_mask"][:]

        return self.handler.get_pixel_mask(
            gap_pixels=self.gap_pixels, double_pixels=self.double_pixels, geometry=self.geometry
        )

    def export(
        self,
        dest: str,
        *,
        disabled_modules: tuple = (),
        index: Iterable | None = None,
        roi: tuple | dict | None = None,
        downsample: tuple | None = None,
        compression: bool = False,
        factor: float | None = None,
        dtype: np.dtype | None = None,
        batch_size: int = 100,
    ) -> None:
        """Export processed data into a separate hdf5 file.

        Args:
            dest (str): Destination hdf5 file path.
            disabled_modules (iterable, optional): Exclude data of provided module indices from
                processing. Defaults to ().
            index (iterable, optional): An iterable with indexes of images to be exported.
                Export all images if None. Defaults to None.
            roi (tuple or dict, optional): A single tuple, or a tuple of tuples with image ROIs in
                a form (bottom, top, left, right), or a dict where each key is a ROI label and
                a corresponding value is a tuple with ROI coordinates. Export whole images if None.
                Defaults to None.
            downsample (tuple, optional): A tuple of 2 integers (N, M). Reduce image size in NxM
                pixel blocks, resulting in an average of valid pixel values within a block.
                Effectively no downsampling is happening in case of (1, 1) or None. Defaults to
                None.
            compression (bool, optional): Apply bitshuffle+lz4 compression. Defaults to False.
            factor (float, optional): If conversion is True, use this factor to divide converted
                values. The output values are also rounded and casted to np.int32 dtype. Keep the
                original values if None. Defaults to None.
            dtype (np.dtype, optional): Resulting image data type. Use dtype of the processed data
                if None. Defaults to None.
            batch_size (int, optional): Process images in batches of that size in order to avoid
                running out of memory. Defaults to 100.
        """
        if self._processed:
            raise RuntimeError("Can not export already processed file.")

        # Deprecated support for boolean downsample parameter (to be removed in ju/4.0)
        if isinstance(downsample, bool):
            warnings.warn(
                "Passing a boolean `downsample` value is deprecated and will be removed in "
                "jungfrau_utils/4.0. Use `(2, 2)` instead of True and `None` instead of False.",
                DeprecationWarning,
            )
            downsample = (2, 2) if downsample else None

        # Deprecated support for dtype parameter (to be removed in ju/4.0)
        if dtype is not None:
            warnings.warn(
                "Passing a `dtype` parameter is deprecated and it will be removed in "
                "jungfrau_utils/4.0. This parameter should not be used anymore as it may cause "
                "issues with hdf5 direct chunk writing with certain `dtype` values. "
                "Setting `dtype` to None.",
                DeprecationWarning,
            )

        if downsample:
            if len(downsample) != 2:
                raise ValueError("Parameter `downsample` must have lenght of 2.")

            if downsample == (1, 1):
                downsample = None

        if roi is not None and downsample:
            raise ValueError("Unsupported mode: roi with downsample.")

        if roi is not None:
            if isinstance(roi, tuple) and len(roi) == 4 and all(isinstance(v, int) for v in roi):
                # this is a single tuple with coordinates, so wrap it in another tuple
                roi = (roi,)

            if isinstance(roi, dict):
                roi_labels = roi.keys()
                roi = tuple(roi.values())
            else:
                roi_labels = range(len(roi))

            out_shape = self.get_shape_out()
            for roi_y1, roi_y2, roi_x1, roi_x2 in roi:
                if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
                    raise ValueError("ROI must have corresponding coordinates in ascending order.")
                if roi_y1 < 0 or roi_y2 >= out_shape[0] or roi_x1 < 0 or roi_x2 >= out_shape[1]:
                    raise ValueError(f"ROI is outside of resulting image size {out_shape}.")

        if index is not None:
            index = np.array(index)  # convert iterable into numpy array

        _mask = self.mask
        if downsample and not _mask:
            warnings.warn("Only masked data can be downsampled, `mask` is set to True")
            self.mask = True

        _factor = self.handler.factor
        if downsample:
            # factor division and rounding should occur after downsampling, so we do it not via
            # JFDataHandler
            self.handler.factor = None
        else:
            self.handler.factor = factor

        _module_map = self.handler.module_map
        if disabled_modules:
            if -1 in self.handler.module_map:
                raise ValueError("Can not disable modules when file contains disabled modules.")

            n_modules = self.handler.detector.n_modules
            if min(disabled_modules) < 0 or max(disabled_modules) >= n_modules:
                raise ValueError(f"Disabled modules must be within 0 {n_modules-1} range.")

            module_map = np.arange(n_modules)
            for ind in sorted(disabled_modules):
                module_map[ind] = -1
                module_map[ind + 1 :] -= 1

            self.handler.module_map = module_map

        # a function for 'visititems' should have the args (name, object)
        def _visititems(name, obj):
            if isinstance(obj, h5py.Dataset):
                if (
                    name == f"data/{self.detector_name}/data"
                    or name.endswith("frame_index")
                    or name == f"data/{self.detector_name}/module_map"
                    or name == f"data/{self.detector_name}/pixel_mask"
                ):
                    return

            if isinstance(obj, h5py.Group):
                h5_dest.create_group(name)

            else:  # isinstance(obj, h5py.Dataset)
                dset_source = self.file[name]

                if name.startswith("data") and not name.endswith("pixel_mask"):
                    # datasets with data per image, so indexing should be applied
                    data = dset_source[:] if index is None else dset_source[index, :]
                    h5_dest.create_dataset(name, data=data)
                else:
                    h5_dest.create_dataset(name, data=dset_source)

            # copy group/dataset attributes (if it's not a dataset with the actual data)
            for key, value in self.file[name].attrs.items():
                h5_dest[name].attrs[key] = value

        with h5py.File(dest, "w") as h5_dest:
            # traverse the source file and copy/index all datasets, except the raw data
            self.file.visititems(_visititems)

            # create `meta` group
            meta_path = f"/data/{self.detector_name}/meta"
            meta_group = h5_dest.create_group(meta_path)

            # save processing parameters in `meta`
            meta_group["module_map"] = self.handler.module_map
            meta_group["conversion"] = self.conversion
            meta_group["mask"] = self.mask
            meta_group["gap_pixels"] = self.gap_pixels
            meta_group["double_pixels"] = self.double_pixels
            meta_group["geometry"] = self.geometry

            meta_group["downsample"] = downsample if downsample else (1, 1)

            # this also sets detector group (channel) as processed
            if self.conversion or self.mask or self.gap_pixels or self.geometry or roi or factor:
                meta_group["conversion_factor"] = factor or np.nan

            # now process the raw data
            dset = self._data_dset
            n_images = dset.shape[0] if index is None else len(index)

            pixel_mask = self.get_pixel_mask()
            out_shape = self.get_shape_out()
            out_dtype = self.get_dtype_out()

            if downsample:
                pixel_mask, good_pixels_fraction = _downsample_mask_jit(pixel_mask, downsample)
                out_shape = tuple(
                    (shape + ds - 1) // ds for shape, ds in zip(out_shape, downsample)
                )
                if factor is not None:
                    out_dtype = np.dtype(np.int32)
            else:
                good_pixels_fraction = pixel_mask.astype(np.float64)

            args = {}
            args["dtype"] = out_dtype
            if compression:
                args.update(compargs)

            if roi is None:
                meta_group["pixel_mask"] = pixel_mask
                meta_group["good_pixels_fraction"] = good_pixels_fraction

                args["shape"] = (n_images, *out_shape)
                args["chunks"] = (1, *out_shape)

                h5_dest.create_dataset(dset.name, **args)

            else:
                # replace the full detector data group with per ROI data groups
                for l, (roi_y1, roi_y2, roi_x1, roi_x2) in zip(roi_labels, roi):
                    h5_dest.copy(
                        source=h5_dest[f"/data/{self.detector_name}"],
                        dest=h5_dest["/data"],
                        name=f"/data/{self.detector_name}:ROI_{l}",
                    )

                    roi_data_group = h5_dest[f"/data/{self.detector_name}:ROI_{l}"]
                    roi_meta_group = roi_data_group["meta"]

                    roi_meta_group["roi"] = [(roi_y1, roi_y2), (roi_x1, roi_x2)]
                    roi_meta_group["pixel_mask"] = pixel_mask[
                        slice(roi_y1, roi_y2), slice(roi_x1, roi_x2)
                    ]

                    roi_shape = (roi_y2 - roi_y1, roi_x2 - roi_x1)

                    args["shape"] = (n_images, *roi_shape)
                    args["chunks"] = (1, *roi_shape)

                    roi_data_group.create_dataset("data", **args)

                del h5_dest[f"/data/{self.detector_name}"]

            # prepare buffers to be reused for every batch
            read_buffer = np.empty((batch_size, *dset.shape[-2:]), dtype=dset.dtype)
            # in case of downsample with a factor, the intermediate out_buffer should have
            # np.float32 dtype
            out_buffer_dtype = np.float32 if downsample and factor is not None else out_dtype
            # shape_out is changed if downsample != (1, 1), so use get_shape_out() once again
            out_buffer = np.zeros((batch_size, *self.get_shape_out()), dtype=out_buffer_dtype)
            if downsample:
                ds_buffer = np.zeros((batch_size, *out_shape), dtype=out_dtype)

            # prepare utility vars
            if downsample and self.parallel:
                downsample_image = _downsample_image_par_jit
            else:
                downsample_image = _downsample_image_jit

            if compression:
                dtype_size = out_dtype.itemsize
                bytes_num_elem = struct.pack(">q", np.prod(out_shape) * dtype_size)
                bytes_block_size = struct.pack(">i", BLOCK_SIZE * dtype_size)
                header = bytes_num_elem + bytes_block_size

            # process and write data in batches
            for batch_start in range(0, n_images, batch_size):
                batch_range = np.arange(batch_start, min(batch_start + batch_size, n_images))
                batch_ind = batch_range if index is None else index[batch_range]

                read_buffer_view = read_buffer[: len(batch_ind)]
                out_buffer_view = out_buffer[: len(batch_ind)]
                if downsample:
                    ds_buffer_view = ds_buffer[: len(batch_ind)]

                # Avoid a stride-bottleneck, see https://github.com/h5py/h5py/issues/977
                if np.sum(np.diff(batch_ind)) == len(batch_ind) - 1:
                    # consecutive index values
                    dset.read_direct(read_buffer_view, source_sel=np.s_[batch_ind])
                else:
                    for i, j in enumerate(batch_ind):
                        dset.read_direct(read_buffer_view, dest_sel=np.s_[i], source_sel=np.s_[j])

                # Process data (`out_buffer_view` is modified in-place)
                self.handler.process(
                    read_buffer_view,
                    conversion=self.conversion,
                    mask=self.mask,
                    gap_pixels=self.gap_pixels,
                    double_pixels=self.double_pixels,
                    geometry=self.geometry,
                    parallel=self.parallel,
                    out=out_buffer_view,
                )

                if downsample:
                    downsample_image(
                        ds_buffer_view, out_buffer_view, downsample, factor, good_pixels_fraction
                    )
                    out_buffer_view = ds_buffer_view

                if roi is None:
                    for pos, im in zip(batch_range, out_buffer_view):
                        if compression:
                            im = header + bitshuffle.compress_lz4(im, BLOCK_SIZE).tobytes()

                        h5_dest[dset.name].id.write_direct_chunk((pos, 0, 0), im)

                else:
                    for l, (roi_y1, roi_y2, roi_x1, roi_x2) in zip(roi_labels, roi):
                        roi_data = out_buffer_view[:, slice(roi_y1, roi_y2), slice(roi_x1, roi_x2)]
                        h5_dest[f"/data/{self.detector_name}:ROI_{l}/data"][batch_range] = roi_data

    def _export_v2(
        self,
        dest: str,
        *,
        disabled_modules: tuple = (),
        index: Iterable | None = None,
        compression: bool = False,
        factor: float | None = None,
        batch_size: int = 100,
        func=None,
    ) -> None:
        """Export processed data into a separate hdf5 file.

        Args:
            dest (str): Destination hdf5 file path.
            disabled_modules (iterable, optional): Exclude data of provided module indices from
                processing. Defaults to ().
            index (iterable, optional): An iterable with indexes of images to be exported.
                Export all images if None. Defaults to None.
            compression (bool, optional): Apply bitshuffle+lz4 compression. Defaults to False.
            factor (float, optional): If conversion is True, use this factor to divide converted
                values. The output values are also rounded and casted to np.int32 dtype. Keep the
                original values if None. Defaults to None.
            batch_size (int, optional): Process images in batches of that size in order to avoid
                running out of memory. Defaults to 100.
            func: TODO
        """
        if self._processed:
            raise RuntimeError("Can not export already processed file.")

        if index is not None:
            index = np.array(index)  # convert iterable into numpy array

        _mask = self.mask

        dic_file = {"": {}}

        _factor = self.handler.factor
        if factor:
            if func:
                # factor division and rounding should occur after function, so we do it not via
                # JFDataHandler
                self.handler.factor = None
                out_dtype = np.dtype(np.int32)
                dic_file[""]["conversion_factor"] = factor or np.nan
            else:
                self.handler.factor = factor

        _module_map = self.handler.module_map
        if disabled_modules:
            if -1 in self.handler.module_map:
                raise ValueError("Can not disable modules when file contains disabled modules.")

            n_modules = self.handler.detector.n_modules
            if min(disabled_modules) < 0 or max(disabled_modules) >= n_modules:
                raise ValueError(f"Disabled modules must be within 0 {n_modules-1} range.")

            module_map = np.arange(n_modules)
            for ind in sorted(disabled_modules):
                module_map[ind] = -1
                module_map[ind + 1 :] -= 1

            self.handler.module_map = module_map

        # a function for 'visititems' should have the args (name, object)
        def _visititems(name, obj):
            if isinstance(obj, h5py.Dataset):
                if (
                    name == f"data/{self.detector_name}/data"
                    or name.endswith("frame_index")
                    or name == f"data/{self.detector_name}/module_map"
                    or name == f"data/{self.detector_name}/pixel_mask"
                ):
                    return

            if isinstance(obj, h5py.Group):
                h5_dest.create_group(name)

            else:  # isinstance(obj, h5py.Dataset)
                dset_source = self.file[name]

                if name.startswith("data") and not name.endswith("pixel_mask"):
                    # datasets with data per image, so indexing should be applied
                    data = dset_source[:] if index is None else dset_source[index, :]
                    h5_dest.create_dataset(name, data=data)
                else:
                    h5_dest.create_dataset(name, data=dset_source)

            # copy group/dataset attributes (if it's not a dataset with the actual data)
            for key, value in self.file[name].attrs.items():
                h5_dest[name].attrs[key] = value

        with h5py.File(dest, "w") as h5_dest:
            # traverse the source file and copy/index all datasets, except the raw data
            self.file.visititems(_visititems)

            # create `meta` group
            meta_path = f"/data/{self.detector_name}/meta"
            meta_group = h5_dest.create_group(meta_path)

            # save processing parameters in `meta`
            meta_group["module_map"] = self.handler.module_map
            meta_group["conversion"] = self.conversion
            meta_group["mask"] = self.mask
            meta_group["gap_pixels"] = self.gap_pixels
            meta_group["double_pixels"] = self.double_pixels
            meta_group["geometry"] = self.geometry

            dset = self._data_dset
            n_images = dset.shape[0] if index is None else len(index)

            pixel_mask = self.get_pixel_mask()
            out_dtype = self.get_dtype_out()

            if func and factor is not None:
                out_dtype = np.dtype(np.int32)

            args = {}
            args["dtype"] = out_dtype
            if compression:
                args.update(compargs)
                header = {}

            dic_file[""]["pixel_mask"] = pixel_mask
            dic_file[""]["downsample"] = (1, 1)

            # prepare buffers to be reused for every batch
            read_buffer = np.empty((batch_size, *dset.shape[-2:]), dtype=dset.dtype)
            out_buffer_dtype = np.float32 if func and factor is not None else out_dtype
            # shape_out is changed if downsample != (1, 1), so use get_shape_out() once again
            out_buffer = np.zeros((batch_size, *self.get_shape_out()), dtype=out_buffer_dtype)

            # process and write data in batches
            for batch_start in range(0, n_images, batch_size):
                batch_range = np.arange(batch_start, min(batch_start + batch_size, n_images))
                batch_ind = batch_range if index is None else index[batch_range]

                read_buffer_view = read_buffer[: len(batch_ind)]
                out_buffer_view = out_buffer[: len(batch_ind)]

                # Avoid a stride-bottleneck, see https://github.com/h5py/h5py/issues/977
                if np.sum(np.diff(batch_ind)) == len(batch_ind) - 1:
                    # consecutive index values
                    dset.read_direct(read_buffer_view, source_sel=np.s_[batch_ind])
                else:
                    for i, j in enumerate(batch_ind):
                        dset.read_direct(read_buffer_view, dest_sel=np.s_[i], source_sel=np.s_[j])

                # Process data (`out_buffer_view` is modified in-place)
                self.handler.process(
                    read_buffer_view,
                    conversion=self.conversion,
                    mask=self.mask,
                    gap_pixels=self.gap_pixels,
                    double_pixels=self.double_pixels,
                    geometry=self.geometry,
                    parallel=self.parallel,
                    out=out_buffer_view,
                )

                # here? compression bit if rois also like this?, 1st batch if needed
                if batch_range[0] == 0:
                    out_shape = {}
                    buf_shape = {}
                    if func:
                        st_frame = func(self, out_buffer_view[0:1, :, :], dic_file, None)
                        func_buffer = {}
                        func_buffer_view = {}
                    else:
                        st_frame = {"": out_buffer_view[0:1, :, :]}
                        dic_file[""]["data"] = out_buffer_view[0:1, :, :]

                    for key in st_frame:
                        if key[0:5] != ":ROI_":
                            continue

                        h5_dest.copy(
                            source=h5_dest[f"/data/{self.detector_name}"],
                            dest=h5_dest["/data"],
                            name=f"/data/{self.detector_name}{key}",
                        )
                    if "" not in st_frame:
                        del h5_dest[f"/data/{self.detector_name}"]

                    for key in st_frame:
                        if not (key == "" or key[0:5] == ":ROI_"):
                            print("Unknown key({key}) for data in buffer, skipped.")
                        tmp = np.shape(dic_file[key]["data"])
                        out_shape[key] = tmp[1:] if len(tmp) > 1 else (1,)
                        tmp = np.shape(st_frame[key])
                        buf_shape[key] = tmp[1:] if len(tmp) > 1 else (1,)
                        if func:
                            func_buffer[key] = np.zeros(
                                (batch_size, *buf_shape[key]), dtype=out_dtype
                            )

                        args["shape"] = (n_images, *out_shape[key])
                        args["chunks"] = (1, *out_shape[key])
                        dset_name = f"data/{self.detector_name}{key}/data"
                        h5_dest.create_dataset(dset_name, **args)
                        if compression:
                            dtype_size = out_dtype.itemsize
                            bytes_num_elem = struct.pack(">q", np.prod(out_shape[key]) * dtype_size)
                            bytes_block_size = struct.pack(">i", BLOCK_SIZE * dtype_size)
                            header[key] = bytes_num_elem + bytes_block_size

                if func:
                    for key in func_buffer:
                        func_buffer_view[key] = func_buffer[key][: len(batch_ind)]
                    func(self, out_buffer_view, dic_file, func_buffer_view)
                    keys_write = func_buffer_view.keys()
                else:
                    dic_file[""]["data"] = out_buffer_view
                    keys_write = [""]
                for label, dic in dic_file.items():
                    if label not in keys_write:
                        continue
                    for key, value in dic.items():
                        if key == "data":
                            if value.dtype != out_dtype:
                                raise TypeError(
                                    f"Data type ({value.dtype}) does not match expected {out_dtype}"
                                )
                            for pos, im in zip(batch_range, value):
                                if len(out_shape[label]) == 2:
                                    loc = (pos, 0, 0)
                                elif len(out_shape[label]) == 1:
                                    loc = (pos, 0)
                                else:
                                    print(
                                        "Unexpected dimensions of data:{len(out_shape[label])}, accepted 1 or 2"
                                    )
                                if compression:
                                    im = (
                                        header[key]
                                        + bitshuffle.compress_lz4(im, BLOCK_SIZE).tobytes()
                                    )
                                dset_name = f"data/{self.detector_name}{label}/data"
                                h5_dest[dset_name].id.write_direct_chunk(loc, im)

                        elif value is not None:
                            h5_dest[f"data/{self.detector_name}{label}/meta/{key}"] = value

        # restore original values
        self.mask = _mask
        self.handler.factor = _factor
        self.handler.module_map = _module_map

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item: str | tuple | int | slice | range | list | NDArray) -> NDArray:
        if isinstance(item, str):
            # per pulse data entry (lazy)
            return self._data_group[item]

        if isinstance(item, tuple):
            # multiple arguments: first is index, the rest is roi
            ind, roi = item[0], item[1:]
        elif isinstance(item, (int, slice, range, list, np.ndarray)):
            # single image index, no roi
            ind, roi = item, ()
        else:
            raise TypeError("Unknown selection type.")

        # Avoid a stride-bottleneck, see https://github.com/h5py/h5py/issues/977
        if isinstance(ind, numbers.Integral):
            is_index_consecutive = True
        elif isinstance(ind, (slice, range)):
            is_index_consecutive = ind.step is None or ind.step == 1
        elif isinstance(ind, (list, tuple, np.ndarray)):
            is_index_consecutive = np.sum(np.diff(ind)) == len(ind) - 1
        else:
            raise TypeError("Unknown index type")

        dset = self._data_dset
        if is_index_consecutive:
            data = dset[ind]
        else:
            if isinstance(ind, slice):
                ind = list(islice(range(dset.shape[0]), ind.start, ind.stop, ind.step))

            data = np.empty(shape=(len(ind), *dset.shape[-2:]), dtype=dset.dtype)
            for i, j in enumerate(ind):
                data[i] = dset[j]

        if self._processed:
            # recover keV values if there was a factor used upon exporting
            conversion_factor = self._meta_group["conversion_factor"]
            if not np.isnan(conversion_factor):
                data = np.multiply(data, conversion_factor, dtype=np.float32)
        else:
            data = self.handler.process(
                data,
                conversion=self.conversion,
                mask=self.mask,
                gap_pixels=self.gap_pixels,
                double_pixels=self.double_pixels,
                geometry=self.geometry,
                parallel=self.parallel,
            )

        if roi:
            if data.ndim == 3:
                roi = (slice(None), *roi)
            data = data[roi]

        return data

    def __repr__(self) -> str:
        if self.file.id:
            r = f'<Jungfrau file "{self.file.filename}">'
        else:
            r = "<Closed Jungfrau file>"
        return r

    def close(self) -> None:
        """Close Jungfrau file."""
        if self.file.id:
            self.file.close()
        del self.handler  # dereference handler since it holds pedestal/gain data

    def __len__(self) -> int:
        return len(self.file)

    def __getattr__(self, name):
        return getattr(self.file, name)


@njit
def _downsample_mask_jit(mask: NDArray, downsample: tuple[int, int]) -> tuple[NDArray, NDArray]:
    size_y, size_x = mask.shape
    ds_y, ds_x = downsample
    downsample_pix_num = ds_y * ds_x
    out_shape_y = (size_y + ds_y - 1) // ds_y
    out_shape_x = (size_x + ds_x - 1) // ds_x

    downsampled_mask = np.zeros(shape=(out_shape_y, out_shape_x), dtype=np.bool_)
    good_pixels_fraction = np.zeros(shape=(out_shape_y, out_shape_x), dtype=np.float64)

    for i1 in range(out_shape_y):
        i_y = ds_y * i1
        for i2 in range(out_shape_x):
            i_x = ds_x * i2
            _mask = mask[i_y : i_y + ds_y, i_x : i_x + ds_x]
            if i_y + ds_y > size_y or i_x + ds_x > size_x:
                # set mask to False for pixels including remainder pixels
                downsampled_mask[i1, i2] = False
            else:
                downsampled_mask[i1, i2] = np.all(_mask)
            good_pixels_fraction[i1, i2] = np.count_nonzero(_mask) / downsample_pix_num

    return downsampled_mask, good_pixels_fraction


def _downsample_image(
    res: NDArray,
    image: NDArray,
    downsample: tuple[int, int],
    factor: float | None,
    good_pixels_fraction: NDArray,
) -> None:
    num, out_shape_y, out_shape_x = res.shape
    ds_y, ds_x = downsample

    for i1 in prange(num):  # pylint: disable=not-an-iterable
        for i2 in range(out_shape_y):
            i_y = ds_y * i2
            for i3 in range(out_shape_x):
                i_x = ds_x * i3

                gpr = good_pixels_fraction[i2, i3]
                tmp_res = np.sum(image[i1, i_y : i_y + ds_y, i_x : i_x + ds_x]) / gpr if gpr else 0

                if factor is None:
                    res[i1, i2, i3] = tmp_res
                else:
                    res[i1, i2, i3] = round(tmp_res / factor)


_downsample_image_jit = njit(_downsample_image)
_downsample_image_par_jit = njit(parallel=True)(_downsample_image)
