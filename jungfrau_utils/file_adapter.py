import os
from pathlib import Path

import h5py
import numpy as np
from bitshuffle.h5 import H5_COMPRESS_LZ4, H5FILTER  # pylint: disable=no-name-in-module

from jungfrau_utils.data_handler import JFDataHandler
from jungfrau_utils.swissfel_helpers import locate_gain_file, locate_pedestal_file

# bitshuffle hdf5 filter params
BLOCK_SIZE = 0
compargs = {"compression": H5FILTER, "compression_opts": (BLOCK_SIZE, H5_COMPRESS_LZ4)}
# limit bitshuffle omp to a single thread
# a better fix would be to use bitshuffle compiled without omp support
os.environ["OMP_NUM_THREADS"] = "1"

BATCH_SIZE = 1000


class File:
    """Jungfrau file"""

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
        """Create a new Jungfrau file wrapper

        Args:
            file_path (str): path to Jungfrau file
            gain_file (str, optional): path to gain file. Auto-locate if empty. Defaults to ''.
            pedestal_file (str, optional): path to pedestal file. Auto-locate if empty.
                Defaults to ''.
            conversion (bool, optional): Apply gain conversion and pedestal correction.
                Defaults to True.
            mask (bool, optional): Perform masking of bad pixels (assign them to 0).
                Defaults to True.
            gap_pixels (bool, optional): Add gap pixels between detector chips.
                Defaults to True.
            geometry (bool, optional): Apply geometry correction. Defaults to True.
            parallel (bool, optional): Use parallelized processing. Defaults to True.
        """
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
        """Detector name (readonly)"""
        return self.handler.detector_name

    @property
    def gain_file(self):
        """Gain file path (readonly)"""
        return self.handler.gain_file

    @property
    def pedestal_file(self):
        """Pedestal file path (readonly)"""
        return self.handler.pedestal_file

    @property
    def conversion(self):
        """A flag for applying pedestal correction and gain conversion"""
        return self._conversion

    @conversion.setter
    def conversion(self, value):
        if self._processed:
            print("The file is already processed, setting 'conversion' to False")
            value = False

        self._conversion = value

    @property
    def mask(self):
        """A flag for masking bad pixels"""
        return self._mask

    @mask.setter
    def mask(self, value):
        if self._processed:
            print("The file is already processed, setting 'mask' to False")
            value = False

        self._mask = value

    @property
    def gap_pixels(self):
        """A flag for adding gap pixels"""
        return self._gap_pixels

    @gap_pixels.setter
    def gap_pixels(self, value):
        if self._processed:
            print("The file is already processed, setting 'gap_pixels' to False")
            value = False

        self._gap_pixels = value

    @property
    def geometry(self):
        """A flag for applying geometry"""
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if self._processed:
            print("The file is already processed, setting 'geometry' to False")
            value = False

        self._geometry = value

    @property
    def parallel(self):
        """A flag for using parallelization"""
        return self._parallel

    @parallel.setter
    def parallel(self, value):
        self._parallel = value

    @property
    def _processed(self):
        # TODO: generalize this check for data reduction case, where dtype can be different
        return self.file[f"/data/{self.detector_name}/data"].dtype == np.float32

    def save_roi(self, dest, roi_x, roi_y, compress=False, factor=None, dtype=None):
        """Save data in a separate hdf5 file

        Args:
            dest (str): Destination file path
            roi_x (tuple): ROIs to save along x axis.
            roi_y (tuple): ROIs to save along y axis.
            compress (bool, optional): Apply bitshuffle+lz4 compression. Defaults to False.
            factor (float, optional): Divide all values by a factor. Defaults to None.
            dtype (np.dtype, optional): Resulting image data type. Defaults to None.
        """

        def copy_objects(name, obj):
            if isinstance(obj, h5py.Group):
                h5_dest.create_group(name)

            elif isinstance(obj, h5py.Dataset):
                dset_source = self.file[name]

                if name == f"data/{self.detector_name}/data":  # compress and copy
                    args = dict()
                    n_images = self["data"].shape[0]

                    h5_dest.create_dataset(f"data/{self.detector_name}/n_roi", data=len(roi_x))

                    for i, (roix, roiy) in enumerate(zip(roi_x, roi_y)):
                        h5_dest.create_dataset(
                            f"data/{self.detector_name}/roi_{i}", data=[roiy, roix]
                        )

                        # save a pixel mask for roi
                        pixel_mask_roi = self.handler.get_pixel_mask(
                            gap_pixels=self.gap_pixels, geometry=self.geometry
                        )
                        h5_dest.create_dataset(
                            f"data/{self.detector_name}/pixel_mask_roi_{i}",
                            data=pixel_mask_roi[slice(*roiy), slice(*roix)],
                        )

                        # prepare ROI datasets
                        roi_shape = (roiy[1] - roiy[0], roix[1] - roix[0])

                        args["shape"] = (n_images, *roi_shape)
                        args["maxshape"] = (n_images, *roi_shape)
                        args["chunks"] = (1, *roi_shape)

                        if dtype is None:
                            args["dtype"] = self.handler.get_dtype_out(
                                self["data"].dtype, conversion=self.conversion
                            )
                        else:
                            args["dtype"] = dtype

                        if compress:
                            args.update(compargs)

                        h5_dest.create_dataset(f"{name}_roi_{i}", **args)

                    for ind in range(0, n_images, BATCH_SIZE):
                        batch_slice = slice(ind, min(ind + BATCH_SIZE, n_images))
                        batch_data = self[batch_slice]

                        for i, (roix, roiy) in enumerate(zip(roi_x, roi_y)):
                            roi_data = batch_data[:, slice(*roiy), slice(*roix)]

                            if factor:
                                roi_data = np.round(roi_data / factor)

                            h5_dest[f"{name}_roi_{i}"][batch_slice] = roi_data

                else:  # copy
                    h5_dest.create_dataset_like(name, dset_source, data=dset_source)

            if name != f"data/{self.detector_name}/data":
                # copy attributes
                for key, value in self.file[name].attrs.items():
                    h5_dest[name].attrs[key] = value

        with h5py.File(dest, "w") as h5_dest:
            self.file.visititems(copy_objects)

    def export_plain_data(self, dest, index=None, compress=False, factor=None, dtype=None):
        """Export data in a separate plain hdf5 file

        Args:
            dest (str): Destination file path
            index (list): List of image indexes to save. Export all images if None.
                Defaults to None.
            compress (bool, optional): Apply bitshuffle+lz4 compression. Defaults to False.
            factor (float, optional): Divide all values by a factor. Defaults to None.
            dtype (np.dtype, optional): Resulting image data type. Defaults to None.
        """
        if index is None:
            index = slice(None, None)
        elif isinstance(index, int):
            index = [index]

        data_group = self.file[f"/data/{self.detector_name}"]

        def export_objects(name):
            dset_source = data_group[name]
            args = dict()

            if name == "data":  # compress and copy
                data = self[index, :, :]
                if factor:
                    data = np.round(data / factor)

                args["shape"] = data.shape
                args["maxshape"] = data.shape
                args["chunks"] = (1, *data.shape[1:])

                if dtype is None:
                    args["dtype"] = data.dtype
                else:
                    args["dtype"] = dtype

                if compress:
                    args.update(compargs)

                dset_dest = h5_dest.create_dataset_like(f"/data/{name}", dset_source, **args)
                dset_dest[:] = data

            else:  # copy
                data = dset_source[index, :]
                args["shape"] = data.shape
                args["maxshape"] = data.shape
                h5_dest.create_dataset_like(f"/data/{name}", dset_source, data=data, **args)

        with h5py.File(dest, "w") as h5_dest:
            h5_dest.create_group("/data")
            data_group.visit(export_objects)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, str):
            # metadata entry (lazy)
            return self.file[f"/data/{self.detector_name}/{item}"]

        elif isinstance(item, (int, slice)):
            # single image index or slice, no roi
            ind, roi = item, ()

        else:
            # image index and roi
            ind, roi = item[0], item[1:]

        data = self.file[f"/data/{self.detector_name}/data"][:][ind]
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
        """Close Jungfrau file"""
        if self.file.id:
            self.file.close()

    def __len__(self):
        return len(self.file)

    def __getattr__(self, name):
        return getattr(self.file, name)
