import os
import struct
import warnings
from functools import partial
from itertools import islice
from pathlib import Path

import bitshuffle
import h5py
import numpy as np
from bitshuffle.h5 import H5_COMPRESS_LZ4, H5FILTER  # pylint: disable=no-name-in-module

from jungfrau_utils.data_handler import JFDataHandler
from jungfrau_utils.swissfel_helpers import locate_gain_file, locate_pedestal_file

warnings.filterwarnings("default", category=DeprecationWarning)

# bitshuffle hdf5 filter params
BLOCK_SIZE = 2048
compargs = {"compression": H5FILTER, "compression_opts": (BLOCK_SIZE, H5_COMPRESS_LZ4)}
# limit bitshuffle omp to a single thread
# a better fix would be to use bitshuffle compiled without omp support
os.environ["OMP_NUM_THREADS"] = "1"


class File:
    """Jungfrau file wrapper.

    Args:
        file_path (str): Path to Jungfrau file
        gain_file (str, optional): Path to gain file. Auto-locate if empty. Defaults to ''.
        pedestal_file (str, optional): Path to pedestal file. Auto-locate if empty. Defaults to ''.
        conversion (bool, optional): Apply gain conversion and pedestal correction.
            Defaults to True.
        mask (bool, optional): Perform masking of bad pixels (assign them to 0). Defaults to True.
        gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
        geometry (bool, optional): Apply geometry correction. Defaults to True.
        parallel (bool, optional): Use parallelized processing. Defaults to True.
    """

    def __init__(
        self,
        file_path,
        gain_file="",
        pedestal_file="",
        conversion=True,
        mask=True,
        gap_pixels=True,
        geometry=True,
        parallel=True,
    ):
        self.file_path = Path(file_path)

        self.file = h5py.File(self.file_path, "r")
        self.handler = JFDataHandler(self.file["/general/detector_name"][()].decode())

        self._conversion = conversion
        self._mask = mask
        self._gap_pixels = gap_pixels
        self._geometry = geometry
        self._parallel = parallel

        # Gain file
        if not gain_file:
            gain_file = locate_gain_file(file_path)

        self.handler.gain_file = gain_file

        # Pedestal file (with a pixel mask)
        if not pedestal_file:
            pedestal_file = locate_pedestal_file(file_path)

        self.handler.pedestal_file = pedestal_file

        if "module_map" in self.file[f"/data/{self.detector_name}"]:
            # Pick only the first row (module_map of the first frame), because it is not expected
            # that module_map ever changes during a run. In fact, it is forseen in the future that
            # this data will be saved as a single row for the whole run.
            module_map = self.file[f"/data/{self.detector_name}/module_map"][0, :]
        else:
            module_map = None

        self.handler.module_map = module_map

        # TODO: Here we use daq_rec only of the first pulse within an hdf5 file, however its
        # value can be different for later pulses and this needs to be taken care of. Currently,
        # _allow_n_images decorator applies a function in a loop, making it impossible to change
        # highgain for separate images in a 3D stack.
        daq_rec = self.file[f"/data/{self.detector_name}/daq_rec"][0]

        self.handler.highgain = daq_rec & 0b1

    @property
    def detector_name(self):
        """Detector name (readonly).
        """
        return self.handler.detector_name

    @property
    def gain_file(self):
        """Gain file path (readonly).
        """
        return self.handler.gain_file

    @property
    def pedestal_file(self):
        """Pedestal file path (readonly).
        """
        return self.handler.pedestal_file

    @property
    def conversion(self):
        """A flag for applying pedestal correction and gain conversion.
        """
        return self._conversion

    @conversion.setter
    def conversion(self, value):
        if self._processed:
            print("The file is already processed, setting 'conversion' to False")
            value = False

        self._conversion = value

    @property
    def mask(self):
        """A flag for masking bad pixels.
        """
        return self._mask

    @mask.setter
    def mask(self, value):
        if self._processed:
            print("The file is already processed, setting 'mask' to False")
            value = False

        self._mask = value

    @property
    def gap_pixels(self):
        """A flag for adding gap pixels.
        """
        return self._gap_pixels

    @gap_pixels.setter
    def gap_pixels(self, value):
        if self._processed:
            print("The file is already processed, setting 'gap_pixels' to False")
            value = False

        self._gap_pixels = value

    @property
    def geometry(self):
        """A flag for applying geometry.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if self._processed:
            print("The file is already processed, setting 'geometry' to False")
            value = False

        self._geometry = value

    @property
    def parallel(self):
        """A flag for using parallelization.
        """
        return self._parallel

    @parallel.setter
    def parallel(self, value):
        self._parallel = value

    @property
    def _processed(self):
        # TODO: generalize this check for data reduction case, where dtype can be different
        return self.file[f"/data/{self.detector_name}/data"].dtype == np.float32

    @property
    def _data_dataset(self):
        return f"data/{self.detector_name}/data"

    def export(
        self,
        dest,
        index=None,
        roi=None,
        compression=False,
        factor=None,
        dtype=None,
        batch_size=1000,
    ):
        """Export processed data into a separate hdf5 file.

        Args:
            dest (str): Destination hdf5 file path.
            index (iterable): An iterable with indexes of images to be exported.
                Export all images if None. Defaults to None.
            roi (tuple): A single tuple, or a tuple of tuples with image ROIs in a form
                (bottom, top, left, right). Export whole images if None. Defaults to None.
            compression (bool, optional): Apply bitshuffle+lz4 compression. Defaults to False.
            factor (float, optional): If conversion is True, use this factor to divide converted
                values. The output values are also rounded and casted to np.int32 dtype. Keep the
                original values if None. Defaults to None.
            dtype (np.dtype, optional): Resulting image data type. Use dtype of the processed data
                if None. Defaults to None.
            batch_size (int, optional): Process images in batches of that size in order to avoid
                running out of memory. Defaults to 1000.
        """
        self.handler.factor = factor

        with h5py.File(dest, "w") as h5_dest:
            # a function for 'visititems' should have the args (name, object)
            self.file.visititems(
                partial(
                    self._visititems,
                    h5_dest=h5_dest,
                    index=index,
                    roi=roi,
                    compression=compression,
                    dtype=dtype,
                    batch_size=batch_size,
                )
            )

    def _visititems(self, name, obj, h5_dest, index, roi, compression, dtype, batch_size):
        if isinstance(obj, h5py.Group):
            h5_dest.create_group(name)

        elif isinstance(obj, h5py.Dataset):
            dset_source = self.file[name]

            if name == self._data_dataset:
                self._process_data(h5_dest, index, roi, compression, dtype, batch_size)

            else:
                if name.startswith("data"):
                    # datasets with data per image, so indexing should be applied
                    if index is None:
                        index = slice(None)
                    data = dset_source[index, :]
                    args = {"shape": data.shape, "maxshape": data.shape}
                    h5_dest.create_dataset_like(name, dset_source, data=data, **args)
                else:
                    h5_dest.create_dataset_like(name, dset_source, data=dset_source)

        else:
            raise TypeError(f"Unknown h5py object type {obj}")

        # copy attributes if it's not a dataset with the actual data
        if name != self._data_dataset:
            for key, value in self.file[name].attrs.items():
                h5_dest[name].attrs[key] = value

    def _process_data(self, h5_dest, index, roi, compression, dtype, batch_size):
        args = dict()
        if index is None:
            n_images = self["data"].shape[0]
        else:
            index = np.array(index)
            n_images = len(index)

        if roi is None:
            image_shape = self.handler.get_shape_out(self.gap_pixels, self.geometry)
            # TODO: this is not ideal, find a way to avoid the 2 next lines
            if self.geometry and self.detector_name.startswith("JF06"):
                image_shape = image_shape[1], image_shape[0]

            args["shape"] = (n_images, *image_shape)
            args["maxshape"] = (n_images, *image_shape)
            args["chunks"] = (1, *image_shape)

            if dtype is None:
                args["dtype"] = self.handler.get_dtype_out(
                    self["data"].dtype, conversion=self.conversion
                )
            else:
                args["dtype"] = dtype

            if compression:
                args.update(compargs)

            h5_dest.create_dataset_like(self._data_dataset, self.file[self._data_dataset], **args)

        else:
            if len(roi) == 4 and all(isinstance(v, int) for v in roi):
                # this is a single tuple with coordinates, so wrap it in another tuple
                roi = (roi,)

            h5_dest.create_dataset(f"data/{self.detector_name}/n_roi", data=len(roi))
            for i, (roi_y1, roi_y2, roi_x1, roi_x2) in enumerate(roi):
                h5_dest.create_dataset(
                    f"data/{self.detector_name}/roi_{i}", data=[(roi_y1, roi_y2), (roi_x1, roi_x2)]
                )

                # save a pixel mask for ROI
                pixel_mask_roi = self.handler.get_pixel_mask(
                    gap_pixels=self.gap_pixels, geometry=self.geometry
                )
                h5_dest.create_dataset(
                    f"data/{self.detector_name}/pixel_mask_roi_{i}",
                    data=pixel_mask_roi[slice(roi_y1, roi_y2), slice(roi_x1, roi_x2)],
                )

                # prepare ROI datasets
                roi_shape = (roi_y2 - roi_y1, roi_x2 - roi_x1)

                args["shape"] = (n_images, *roi_shape)
                args["maxshape"] = (n_images, *roi_shape)
                args["chunks"] = (1, *roi_shape)

                if dtype is None:
                    args["dtype"] = self.handler.get_dtype_out(
                        self["data"].dtype, conversion=self.conversion
                    )
                else:
                    args["dtype"] = dtype

                if compression:
                    args.update(compargs)

                h5_dest.create_dataset(f"{self._data_dataset}_roi_{i}", **args)

        dset = self.file[f"/data/{self.detector_name}/data"]

        # process and write data in batches
        for ind in range(0, n_images, batch_size):
            batch_range = range(ind, min(ind + batch_size, n_images))

            if index is None:
                batch_ind = batch_range
            else:
                batch_ind = index[batch_range]

            if np.sum(np.diff(batch_ind)) == len(batch_ind) - 1:
                # consecutive index values
                batch_data = dset[batch_ind]
            else:
                batch_data = np.empty(shape=(len(batch_ind), *dset.shape[1:]), dtype=dset.dtype)
                for i, j in enumerate(batch_ind):
                    batch_data[i] = dset[j]

            # Process data
            batch_data = self.handler.process(
                batch_data,
                conversion=self.conversion,
                mask=self.mask,
                gap_pixels=self.gap_pixels,
                geometry=self.geometry,
                parallel=self.parallel,
            )

            if roi is None:
                bytes_number_of_elements = struct.pack(">q", image_shape[0] * image_shape[1] * 4)
                bytes_block_size = struct.pack(">i", BLOCK_SIZE * 4)
                header = bytes_number_of_elements + bytes_block_size

                for i, im in enumerate(batch_data):
                    compressed = bitshuffle.compress_lz4(im, BLOCK_SIZE)
                    byte_array = header + compressed.tobytes()
                    h5_dest[self._data_dataset].id.write_direct_chunk((ind + i, 0, 0), byte_array)

            else:
                for i, (roi_y1, roi_y2, roi_x1, roi_x2) in enumerate(roi):
                    roi_data = batch_data[:, slice(roi_y1, roi_y2), slice(roi_x1, roi_x2)]
                    h5_dest[f"{self._data_dataset}_roi_{i}"][batch_range] = roi_data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, str):
            # metadata entry (lazy)
            return self.file[f"/data/{self.detector_name}/{item}"]

        if isinstance(item, tuple):
            # multiple arguments: first is index, the rest is roi
            ind, roi = item[0], item[1:]
        elif isinstance(item, (int, slice, range, list, np.ndarray)):
            # single image index, no roi
            ind, roi = item, ()
        else:
            raise TypeError("Unknown selection type.")

        # Avoid a stride-bottleneck, see https://github.com/h5py/h5py/issues/977
        if isinstance(ind, int):
            is_index_consecutive = True
        elif isinstance(ind, (slice, range)):
            is_index_consecutive = ind.step is None or ind.step == 1
        elif isinstance(ind, (list, tuple, np.ndarray)):
            is_index_consecutive = np.sum(np.diff(ind)) == len(ind) - 1

        dset = self.file[f"/data/{self.detector_name}/data"]
        if is_index_consecutive:
            data = dset[ind]
        else:
            if isinstance(ind, slice):
                ind = list(islice(range(dset.shape[0]), ind.start, ind.stop, ind.step))

            data = np.empty(shape=(len(ind), *dset.shape[1:]), dtype=dset.dtype)
            for i, j in enumerate(ind):
                data[i] = dset[j]

        # Process data
        data = self.handler.process(
            data,
            conversion=self.conversion,
            mask=self.mask,
            gap_pixels=self.gap_pixels,
            geometry=self.geometry,
            parallel=self.parallel,
        )

        if roi:
            if data.ndim == 3:
                roi = (slice(None), *roi)
            data = data[roi]

        return data

    def __repr__(self):
        if self.file.id:
            r = f'<Jungfrau file "{self.file_path.name}">'
        else:
            r = "<Closed Jungfrau file>"
        return r

    def close(self):
        """Close Jungfrau file.
        """
        if self.file.id:
            self.file.close()

    def __len__(self):
        return len(self.file)

    def __getattr__(self, name):
        return getattr(self.file, name)
