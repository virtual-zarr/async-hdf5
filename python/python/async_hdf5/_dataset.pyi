from typing import Any

from ._chunk_index import ChunkIndex

class HDF5Dataset:
    """An HDF5 dataset — a typed, shaped, optionally chunked array."""

    @property
    def name(self) -> str:
        """The dataset name."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the dataset in array elements."""

    @property
    def ndim(self) -> int:
        """The number of dimensions."""

    @property
    def dtype(self) -> str:
        """A debug representation of the HDF5 datatype."""

    @property
    def numpy_dtype(self) -> str:
        """Numpy-compatible dtype string (e.g. ``"<f4"``, ``">i8"``, ``"<c16"``).

        Raises:
            ValueError: For datatypes that cannot be represented as a fixed
                numpy dtype (variable-length strings, references).
        """

    @property
    def element_size(self) -> int:
        """The size of a single element in bytes."""

    @property
    def chunk_shape(self) -> tuple[int, ...] | None:
        """The chunk shape, or ``None`` for contiguous/compact datasets."""

    @property
    def filters(self) -> list[dict[str, Any]]:
        """The HDF5 filter pipeline.

        Each filter is a dict with keys ``"id"`` (int), ``"name"`` (str),
        and ``"client_data"`` (list of int).
        """

    @property
    def fill_value(self) -> list[int] | None:
        """Raw fill value as a list of byte values, or ``None`` if not set."""

    async def chunk_index(self) -> ChunkIndex:
        """Build and return the chunk index for this dataset.

        The chunk index maps chunk grid coordinates to byte ranges in the
        file. For chunked datasets this parses the B-tree, fixed array, or
        extensible array on first call.

        Returns:
            The chunk index.
        """

    async def batch_get_chunks(
        self,
        chunk_indices: list[list[int]],
    ) -> list[bytes | None]:
        """Fetch multiple chunks in a single batched I/O call.

        Args:
            chunk_indices: A list of chunk grid coordinate lists.

        Returns:
            A list of raw chunk bytes (or ``None`` for missing chunks),
            in the same order as the input.
        """

    async def batch_fetch_ranges(
        self,
        ranges: list[tuple[int, int]],
    ) -> list[bytes]:
        """Fetch multiple byte ranges in a single batched I/O call.

        No chunk index lookup is performed — the caller must supply
        pre-resolved byte ranges.

        Args:
            ranges: A list of ``(offset, length)`` tuples.

        Returns:
            A list of raw bytes in the same order as the input.
        """

    async def attributes(self) -> dict[str, Any]:
        """Read all attributes on this dataset.

        Returns:
            A dict mapping attribute names to decoded Python values.
        """
