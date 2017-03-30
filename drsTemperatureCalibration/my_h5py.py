import h5py


def create_my_dataset(file, name, shape=None, maxshape=None, dtype=None, **kwargs):
    if maxshape is None:
        maxshape = tuple(x if x != 0 else None for x in shape)


    return file.create_dataset(
        name,
        shape,
        dtype=dtype,
        maxshape=maxshape,
        compression="gzip",
        compression_opts=9,
        fletcher32=True,
        **kwargs
    )

h5py.File.create_my_dataset = create_my_dataset