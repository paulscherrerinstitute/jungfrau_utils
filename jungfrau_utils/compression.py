import h5py
import numpy as np

# see http://python-blosc.blosc.org/reference.html for meaning of numbers
shuffles = ('noshuffle', 'shuffle', 'bitshuffle')
compressors = ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd')

complevel = 9
shuffle = shuffles.index('bitshuffle')
complib = compressors.index('lz4')

compargs = {
    'compression': 32001,
    'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib),
}

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

            if name == dataset:  # compress and copy
                if dtype is None:
                    args_in = compargs
                else:
                    args_in = {**compargs, 'dtype': dtype}

                dset_dest = h5_dest.create_dataset_like(name, dset_source, **args_in)

                # avoid loading the whole dataset in memory
                for i, data in enumerate(dset_source):
                    if factor:
                        data = np.round(data / factor)

                    dset_dest[i] = data

            else:  # copy
                h5_dest.create_dataset_like(name, dset_source, data=dset_source)

        # copy attributes
        for key, value in h5_source[name].attrs.items():
            h5_dest[name].attrs[key] = value

    with h5py.File(f_source, 'r') as h5_source, h5py.File(f_dest, 'w') as h5_dest:
        h5_source.visititems(copy_objects)
