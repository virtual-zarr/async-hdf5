from typing import Any

from ._dataset import HDF5Dataset

class HDF5Group:
    """An HDF5 group — a container for datasets and other groups."""

    @property
    def name(self) -> str:
        """The name of this group."""

    async def children(self) -> list[str]:
        """List the names of all immediate children.

        Returns:
            A list of child names.
        """

    async def group(self, name: str) -> HDF5Group:
        """Open a child group by name.

        Args:
            name: The name of the child group.

        Returns:
            The child group.

        Raises:
            RuntimeError: If the child does not exist or is not a group.
        """

    async def dataset(self, name: str) -> HDF5Dataset:
        """Open a child dataset by name.

        Args:
            name: The name of the child dataset.

        Returns:
            The child dataset.

        Raises:
            RuntimeError: If the child does not exist or is not a dataset.
        """

    async def navigate(self, path: str) -> HDF5Group:
        """Navigate to a group by ``/``-separated path.

        Args:
            path: A ``/``-separated path relative to this group
                (e.g. ``"level1/level2"``).

        Returns:
            The group at the given path.

        Raises:
            RuntimeError: If any component along the path is not found.
        """

    async def group_names(self) -> list[str]:
        """List the names of child groups.

        Returns:
            A list of child group names.
        """

    async def dataset_names(self) -> list[str]:
        """List the names of child datasets.

        Returns:
            A list of child dataset names.
        """

    async def attributes(self) -> dict[str, Any]:
        """Read all attributes on this group.

        Returns:
            A dict mapping attribute names to decoded Python values.
        """
