from ._group import HDF5Group
from ._input import ObspecInput
from .store import ObjectStore

class HDF5File:
    """An open HDF5 file.

    Use [`open`][async_hdf5.HDF5File.open] to create an instance.
    """

    @classmethod
    async def open(
        cls,
        path: str,
        *,
        store: ObjectStore | ObspecInput,
        block_size: int = 8_388_608,
        pre_warm_threshold: int | None = None,
    ) -> HDF5File:
        """Open an HDF5 file.

        Args:
            path: The path within the store to the HDF5 file.
            store: The storage backend (e.g. an obstore ``S3Store``,
                ``LocalStore``, or any object implementing the
                [ObspecInput][async_hdf5.ObspecInput] protocol).
            block_size: The read-ahead block size in bytes for the internal
                block cache. Defaults to 8 MiB.
            pre_warm_threshold: If set, eagerly fetches this many bytes from
                the start of the file on open. Useful for small files where
                the entire file fits in a single read.

        Returns:
            An open HDF5File.
        """

    async def root_group(self) -> HDF5Group:
        """Get the root group of the file.

        Returns:
            The root HDF5 group.
        """
