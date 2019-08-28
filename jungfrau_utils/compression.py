import h5py
import numpy as np
from bitshuffle.h5 import H5FILTER, H5_COMPRESS_LZ4  # pylint: disable=no-name-in-module


BLOCK_SIZE = 0
compargs = {'compression': H5FILTER, 'compression_opts': (BLOCK_SIZE, H5_COMPRESS_LZ4)}


def compress_dataset(f_source, f_dest, dataset, factor=None, dtype=None):
    """Create a new file with a specified dataset being compressed.

    Args:
        f_source: Source file path
        f_dest: Destination file path
        dataset: Dataset to compress
        factor (optional): Defaults to None. If present, the compressed dataset values are:
            compressed_value = np.round(original_value / factor)
        dtype (optional): Defaults to None. Resulting data type for the compressed dataset
    """

    def copy_objects(name, obj):
        if isinstance(obj, h5py.Group):
            h5_dest.create_group(name)

        elif isinstance(obj, h5py.Dataset):
            dset_source = h5_source[name]

            args = {
                k: getattr(dset_source, k)
                for k in (
                    'shape',
                    'dtype',
                    'chunks',
                    'compression',
                    'compression_opts',
                    'scaleoffset',
                    'shuffle',
                    'fletcher32',
                    'fillvalue',
                )
            }

            if dset_source.shape != dset_source.maxshape:
                args['maxshape'] = dset_source.maxshape

            if name == dataset:  # compress and copy
                if dtype is not None:
                    args['dtype'] = dtype

                args.update(compargs)

                dset_dest = h5_dest.create_dataset(name, **args)

                # avoid loading the whole dataset in memory
                for i, data in enumerate(dset_source):
                    if factor:
                        data = np.round(data / factor)

                    dset_dest[i] = data

            else:  # copy
                h5_dest.create_dataset(name, data=dset_source, **args)

        # copy attributes
        for key, value in h5_source[name].attrs.items():
            h5_dest[name].attrs[key] = value

    with h5py.File(f_source, 'r') as h5_source, h5py.File(f_dest, 'w') as h5_dest:
        h5_source.visititems(copy_objects)
