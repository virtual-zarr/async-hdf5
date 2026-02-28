from typing import Iterator

class ChunkLocation:
    """A single chunk's location within the HDF5 file."""

    @property
    def indices(self) -> tuple[int, ...]:
        """Chunk grid coordinates."""

    @property
    def byte_offset(self) -> int:
        """Byte offset of the chunk in the file."""

    @property
    def byte_length(self) -> int:
        """Size of the chunk in bytes (on-disk/compressed size)."""

    @property
    def filter_mask(self) -> int:
        """Bitmask indicating which filters were *not* applied.

        A value of 0 means all filters in the pipeline were applied.
        """

class ChunkIndex:
    """An index mapping chunk grid coordinates to file byte ranges.

    Iterable over [`ChunkLocation`][async_hdf5.ChunkLocation] objects.
    """

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ChunkLocation]: ...

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """The shape of the chunk grid (number of chunks per dimension)."""

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """The chunk shape in array elements."""

    @property
    def dataset_shape(self) -> tuple[int, ...]:
        """The full dataset shape in array elements."""

    def get(self, indices: list[int]) -> ChunkLocation | None:
        """Look up a single chunk by its grid coordinates.

        Args:
            indices: The chunk grid coordinates.

        Returns:
            The chunk location, or ``None`` if the chunk is not allocated.
        """
