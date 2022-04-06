import os
import struct
import warnings
import numbers
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
        double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
            "keep", "mask", or "interp". Defaults to "keep".
        geometry (bool, optional): Apply geometry correction. Defaults to True.
        parallel (bool, optional): Use parallelized processing. Defaults to True.
    """

    def __init__(
        self,
        file_path,
        *,
        gain_file="",
        pedestal_file="",
        conversion=True,
        mask=True,
        gap_pixels=True,
        double_pixels="keep",
        geometry=True,
        parallel=True,
    ):
        self.file_path = Path(file_path)

        self.file = h5py.File(self.file_path, "r")
        self.handler = JFDataHandler(self.file["general/detector_name"][()].decode())

        self._conversion = conversion
        self._mask = mask
        self._gap_pixels = gap_pixels
        self._double_pixels = double_pixels
        self._geometry = geometry
        self._parallel = parallel

        # No need for any further setup if the file is already processed
        if self._processed:
            return

        # Gain file
        if not gain_file:
            gain_file = locate_gain_file(file_path)

        self.handler.gain_file = gain_file

        # Pedestal file (with a pixel mask)
        if not pedestal_file:
            pedestal_file = locate_pedestal_file(file_path)

        self.handler.pedestal_file = pedestal_file

        if "module_map" in self.file[f"data/{self.detector_name}"]:
            # Pick only the first row (module_map of the first frame), because it is not expected
            # that module_map ever changes during a run. In fact, it is forseen in the future that
            # this data will be saved as a single row for the whole run.
            module_map = self.file[f"data/{self.detector_name}/module_map"][0, :]
        else:
            module_map = None

        self.handler.module_map = module_map

        # TODO: Here we use daq_rec only of the first pulse within an hdf5 file, however its
        # value can be different for later pulses and this needs to be taken care of. Currently,
        # _allow_n_images decorator applies a function in a loop, making it impossible to change
        # highgain for separate images in a 3D stack.
        daq_rec = self.file[f"data/{self.detector_name}/daq_rec"][0]

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
            print("The file is already processed, setting 'conversion' has no effect.")
            return

        self._conversion = value

    @property
    def mask(self):
        """A flag for masking bad pixels.
        """
        return self._mask

    @mask.setter
    def mask(self, value):
        if self._processed:
            print("The file is already processed, setting 'mask' has no effect.")
            return

        self._mask = value

    @property
    def gap_pixels(self):
        """A flag for adding gap pixels.
        """
        return self._gap_pixels

    @gap_pixels.setter
    def gap_pixels(self, value):
        if self._processed:
            print("The file is already processed, setting 'gap_pixels' has no effect.")
            return

        self._gap_pixels = value

    @property
    def double_pixels(self):
        """A parameter for making modifications to double pixels.
        """
        return self._double_pixels

    @double_pixels.setter
    def double_pixels(self, value):
        if self._processed:
            print("The file is already processed, setting 'double_pixels' has no effect.")
            return

        self._double_pixels = value

    @property
    def geometry(self):
        """A flag for applying geometry.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if self._processed:
            print("The file is already processed, setting 'geometry' has no effect.")
            return

        self._geometry = value

    @property
    def parallel(self):
        """A flag for using parallelization.
        """
        return self._parallel

    @parallel.setter
    def parallel(self, value):
        if self._processed:
            print("The file is already processed, setting 'parallel' has no effect.")
            return

        self._parallel = value

    @property
    def _processed(self):
        return f"data/{self.detector_name}/conversion_factor" in self.file

    @property
    def _data_dset_name(self):
        return f"data/{self.detector_name}/data"

    def get_shape_out(self):
        """Return the final image shape of a detector, based on gap_pixel and geometry flags.

        Returns:
            tuple: Height and width of a resulting image.
        """
        return self.handler.get_shape_out(gap_pixels=self.gap_pixels, geometry=self.geometry)

    def get_dtype_out(self):
        """Return resulting image dtype of a detector.

        Returns:
            dtype: dtype of a resulting image.
        """
        return self.handler.get_dtype_out(
            self.file[self._data_dset_name].dtype, conversion=self.conversion
        )

    def get_pixel_mask(self):
        """Return pixel mask, shaped according to gap_pixel and geometry flags.

        Returns:
            ndarray: Resulting pixel mask, where True values correspond to valid pixels.
        """
        return self.handler.get_pixel_mask(
            gap_pixels=self.gap_pixels, double_pixels=self.double_pixels, geometry=self.geometry
        )

    def export(
        self,
        dest,
        *,
        index=None,
        roi=None,
        compression=False,
        factor=None,
        dtype=None,
        batch_size=100,
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
                running out of memory. Defaults to 100.
        """
        if self._processed:
            raise RuntimeError("Can not run export, the file is already processed.")

        if index is not None:
            index = np.array(index)  # convert iterable into numpy array

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

            if name == self._data_dset_name:
                self._process_data(h5_dest, index, roi, compression, dtype, batch_size)
            else:
                if name.startswith("data"):
                    # datasets with data per image, so indexing should be applied
                    if index is None:
                        data = dset_source[:, :]
                    else:
                        data = dset_source[index, :]
                    args = {"shape": data.shape, "maxshape": data.shape}
                    h5_dest.create_dataset_like(name, dset_source, data=data, **args)
                else:
                    h5_dest.create_dataset_like(name, dset_source, data=dset_source)

        else:
            raise TypeError(f"Unknown h5py object type {obj}")

        # copy attributes if it's not a dataset with the actual data
        if name != self._data_dset_name:
            for key, value in self.file[name].attrs.items():
                h5_dest[name].attrs[key] = value

    def _process_data(self, h5_dest, index, roi, compression, dtype, batch_size):
        args = dict()

        data_dset = self.file[self._data_dset_name]
        if index is None:
            n_images = data_dset.shape[0]
        else:
            n_images = len(index)

        h5_dest[f"data/{self.detector_name}/conversion_factor"] = self.handler.factor or np.NaN

        pixel_mask = self.get_pixel_mask()

        if roi is None:
            # save a pixel mask
            h5_dest[f"data/{self.detector_name}/pixel_mask"] = pixel_mask

            image_shape = self.get_shape_out()

            args["shape"] = (n_images, *image_shape)
            args["maxshape"] = (n_images, *image_shape)
            args["chunks"] = (1, *image_shape)

            if dtype is None:
                args["dtype"] = self.get_dtype_out()
            else:
                args["dtype"] = dtype

            if compression:
                args.update(compargs)

            h5_dest.create_dataset_like(data_dset.name, data_dset, **args)

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
                h5_dest.create_dataset(
                    f"data/{self.detector_name}/pixel_mask_roi_{i}",
                    data=pixel_mask[slice(roi_y1, roi_y2), slice(roi_x1, roi_x2)],
                )

                # prepare ROI datasets
                roi_shape = (roi_y2 - roi_y1, roi_x2 - roi_x1)

                args["shape"] = (n_images, *roi_shape)
                args["maxshape"] = (n_images, *roi_shape)
                args["chunks"] = (1, *roi_shape)

                if dtype is None:
                    args["dtype"] = self.get_dtype_out()
                else:
                    args["dtype"] = dtype

                if compression:
                    args.update(compargs)

                h5_dest.create_dataset(f"{data_dset.name}_roi_{i}", **args)

        # prepare buffers to be reused for every batch
        read_buffer = np.empty((batch_size, *data_dset.shape[1:]), dtype=data_dset.dtype)

        out_shape = self.get_shape_out()
        out_dtype = self.get_dtype_out()
        out_buffer = np.zeros((batch_size, *out_shape), dtype=out_dtype)

        # process and write data in batches
        for batch_start_ind in range(0, n_images, batch_size):
            batch_range = np.arange(batch_start_ind, min(batch_start_ind + batch_size, n_images))

            if index is None:
                batch_ind = batch_range
            else:
                batch_ind = index[batch_range]

            read_buffer_view = read_buffer[: len(batch_ind)]
            out_buffer_view = out_buffer[: len(batch_ind)]

            # Avoid a stride-bottleneck, see https://github.com/h5py/h5py/issues/977
            if np.sum(np.diff(batch_ind)) == len(batch_ind) - 1:
                # consecutive index values
                data_dset.read_direct(read_buffer_view, source_sel=np.s_[batch_ind])
            else:
                for i, j in enumerate(batch_ind):
                    data_dset.read_direct(read_buffer_view, source_sel=np.s_[j], dest_sel=np.s_[i])

            # Process data
            out_buffer_view = self.handler.process(
                read_buffer_view,
                conversion=self.conversion,
                mask=self.mask,
                gap_pixels=self.gap_pixels,
                double_pixels=self.double_pixels,
                geometry=self.geometry,
                parallel=self.parallel,
                out=out_buffer_view,
            )

            out_buffer_view = np.ascontiguousarray(out_buffer_view)

            if roi is None:
                dtype_size = out_dtype.itemsize
                bytes_num_elem = struct.pack(">q", image_shape[0] * image_shape[1] * dtype_size)
                bytes_block_size = struct.pack(">i", BLOCK_SIZE * dtype_size)
                header = bytes_num_elem + bytes_block_size

                for pos, im in zip(batch_range, out_buffer_view):
                    if compression:
                        byte_array = header + bitshuffle.compress_lz4(im, BLOCK_SIZE).tobytes()
                    else:
                        byte_array = im.tobytes()

                    h5_dest[data_dset.name].id.write_direct_chunk((pos, 0, 0), byte_array)

            else:
                for i, (roi_y1, roi_y2, roi_x1, roi_x2) in enumerate(roi):
                    roi_data = out_buffer_view[:, slice(roi_y1, roi_y2), slice(roi_x1, roi_x2)]
                    h5_dest[f"{data_dset.name}_roi_{i}"][batch_range] = roi_data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, str):
            # metadata entry (lazy)
            return self.file[f"data/{self.detector_name}/{item}"]

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

        dset = self.file[self._data_dset_name]
        if is_index_consecutive:
            data = dset[ind]
        else:
            if isinstance(ind, slice):
                ind = list(islice(range(dset.shape[0]), ind.start, ind.stop, ind.step))

            data = np.empty(shape=(len(ind), *dset.shape[1:]), dtype=dset.dtype)
            for i, j in enumerate(ind):
                data[i] = dset[j]

        if self._processed:
            # recover keV values if there was a factor used upon exporting
            conversion_factor = self.file[f"data/{self.detector_name}/conversion_factor"]
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
        self.handler = None # dereference handler since it holds pedestal/gain data

    def __len__(self):
        return len(self.file)

    def __getattr__(self, name):
        return getattr(self.file, name)
