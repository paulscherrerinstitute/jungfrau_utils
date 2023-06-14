import argparse
from time import sleep, time

import h5py
import numpy


def binary_init(args):
    file = open(args.output_file, "wb")
    return file, file


def binary_write_data(index, writer, data, metadata, n_metadata):
    writer.write(data)

    for n_metadata_dataset in range(n_metadata):
        writer.write(metadata)


def hdf5_init(args):
    file = h5py.File(args.output_file, "w")

    frame_size = [512, 1024]
    writer = {
        "data": file.create_dataset(
            name="data",
            shape=[args.n_frames] + frame_size,
            maxshape=[None] + frame_size,
            dtype="uint16",
        )
    }

    for n_metadata_dataset in range(args.n_metadata):
        dataset_name = "metadata" + str(n_metadata_dataset)
        writer[dataset_name] = file.create_dataset(
            name=dataset_name, shape=[args.n_frames, 1], maxshape=[None, 1], dtype="uint32"
        )

    return file, writer


def hdf5_write_data(index, writer, data, metadata, n_metadata):
    writer["data"][index] = data

    for n_metadata_dataset in range(n_metadata):
        dataset_name = "metadata" + str(n_metadata_dataset)
        writer[dataset_name][index] = metadata


def hdf5_chunked_init(args):
    file = h5py.File(args.output_file, "w")

    frame_size = [512, 1024]
    writer = {
        "data": file.create_dataset(
            name="data",
            shape=[args.n_frames] + frame_size,
            maxshape=[None] + frame_size,
            chunks=tuple([1] + frame_size),
            dtype="uint16",
        )
    }

    for n_metadata_dataset in range(args.n_metadata):
        dataset_name = "metadata" + str(n_metadata_dataset)
        writer[dataset_name] = file.create_dataset(
            name=dataset_name,
            shape=[args.n_frames, 1],
            maxshape=[None, 1],
            chunks=(1, 1),
            dtype="uint32",
        )

    return file, writer


def hdf5_chunked_write_data(index, writer, data, metadata, n_metadata):
    writer["data"].id.write_direct_chunk((index, 0, 0), data)

    for n_metadata_dataset in range(n_metadata):
        dataset_name = "metadata" + str(n_metadata_dataset)
        writer[dataset_name].id.write_direct_chunk((index, 0), metadata)


def main():
    parser = argparse.ArgumentParser(description="GPFS performance tests")
    parser.add_argument("mode", choices=["binary", "hdf5", "hdf5_chunked"])
    parser.add_argument("output_file", type=str)
    parser.add_argument("n_frames", type=int)
    parser.add_argument("n_modules", type=int)
    parser.add_argument("frame_rate", type=int)
    parser.add_argument("n_metadata", type=int)

    arguments = parser.parse_args()

    output_format = (
        "%s\t%d\t%d\t%d\t%d"
        % (
            arguments.mode,
            arguments.n_frames,
            arguments.n_modules,
            arguments.frame_rate,
            arguments.n_metadata,
        )
        + "\t%d\t%f\t%f"
    )

    image_buffer_size_bytes = arguments.n_modules * 512 * 1024 * 2
    metadata_buffer_size_bytes = arguments.n_modules * 4

    if arguments.mode == "binary":

        image_buffer = numpy.random.bytes(image_buffer_size_bytes)
        metadata_buffer = numpy.random.bytes(metadata_buffer_size_bytes)

        write_function = binary_write_data
        file, writer = binary_init(arguments)

    elif arguments.mode == "hdf5":

        image_buffer = numpy.ones(shape=[512, 1024], dtype="uint16")
        metadata_buffer = numpy.ones(shape=[1], dtype="uint32")

        write_function = hdf5_write_data
        file, writer = hdf5_init(arguments)

    elif arguments.mode == "hdf5_chunked":

        image_buffer = numpy.random.bytes(image_buffer_size_bytes)
        metadata_buffer = numpy.random.bytes(metadata_buffer_size_bytes)

        write_function = hdf5_chunked_write_data
        file, writer = hdf5_chunked_init(arguments)

    else:
        raise Exception("Unknown mode %s." % arguments.mode)

    write_function(0, writer, image_buffer, metadata_buffer, arguments.n_metadata)

    start_time_frame = time()

    for index in range(0, arguments.n_frames):
        write_function(index, writer, image_buffer, metadata_buffer, arguments.n_metadata)

        time_to_write = time() - start_time_frame

        time_to_sleep = (1 / arguments.frame_rate) - time_to_write

        if time_to_sleep > 0:
            sleep(time_to_sleep)

        print(output_format % (index, time_to_write, time_to_sleep))

        start_time_frame = time()

    file.close()


if __name__ == "__main__":
    main()
