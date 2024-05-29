from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from multiprocessing.shared_memory import SharedMemory
from typing import Union, Tuple, List
from warnings import warn
from itertools import product
import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike
from overrides import overrides


@dataclass
class LatInfo:
    lattice_vector: str
    dimensions: str
    sites: str
    ids: str
    bounds: str
    adjacency_matrix: str
    lat_vector_type: DTypeLike
    dimensions_type: DTypeLike
    sites_type: DTypeLike
    ids_type: DTypeLike
    bounds_type: DTypeLike
    adjacency_matrix_type: DTypeLike
    lat_shape: Tuple[int, ...]
    dimensions_shape: Tuple[int, ...]
    sites_shape: Tuple[int, ...]
    ids_shape: Tuple[int, ...]
    bounds_shape: Tuple[int, ...]
    adjacency_matrix_shape: Tuple[int, ...]
    num_sites: int
    lat_type: Union[int, None]

    @property
    def all(self) -> list:
        return [self.lattice_vector, self.dimensions, self.sites, self.ids, self.bounds, self.adjacency_matrix]

    @property
    def all_types(self) -> list:
        return [val for key, val in self.__dict__.items() if '_type' in key]

    @property
    def all_shapes(self) -> list:
        return [val for key, val in self.__dict__.items() if '_shape' in key]


@dataclass
class LatSharedMemory:
    lattice_vector: SharedMemory
    dimensions: SharedMemory
    sites: SharedMemory
    ids: SharedMemory
    bounds: SharedMemory
    adjacency_matrix: Union[SharedMemory, None]

    @property
    def all(self) -> list:
        return list(self.__dict__.values())

    @property
    def all_actual_names(self) -> list:
        return list(self.__dict__.keys())


@dataclass
class Lattice:
    """
    Generic crystal lattice class
    """

    dimensions: np.ndarray
    lattice_vector: np.ndarray
    sites: np.ndarray
    ids: np.ndarray
    bounds: np.ndarray
    num_sites: int
    coordination_number: int
    adjacency_matrix: Union[np.ndarray, None]
    dump_cache: Union[Tuple[str, str], None]
    shared_memory: Union[LatSharedMemory, None]
    original: bool

    def get_lattice_slice(self, x: float, margin: float = 0.1) -> np.ndarray:
        """
        Get a slice of the lattice at x
        :param margin: Margin around x
        :param x: x coordinate of the slice
        :return: Slice of the lattice
        """
        return np.array([self.sites[i] for i in range(len(self.sites)) if x - margin < self.sites[i][0] < x + margin])

    def dump(self, file, timestep: Union[int, Tuple[int, float]] = (0, 0.0), suppress_warning: bool = False,
             atom_id: int = None, additional_labels: Union[None, str] = None,
             additional_coords: Union[np.ndarray, List, None] = None) -> None:

        """
        Dump the lattice to a file
        :param additional_coords: Additional coordinates to be dumped after the regular coordinates
        :param additional_labels: a string that represents additional labels to be right after the coordinates
        :param file: File-like object
        :param timestep: Timestep of the dump file (if a tuple is passed, the first element is the timestep and the
        second is the time)
        :param suppress_warning: If true, a warning is not raised if the lattice has more than 3 dimensions
        :param atom_id: If provided, will only dump a single site
        :return: None
        """
        if len(self.dimensions) > 3 and not suppress_warning:
            warn("Dumping lattices with more than 3 dimensions is not standard")
        if isinstance(timestep, tuple):
            timestep = ' '.join(map(str, timestep))

        if atom_id is not None:
            ids = [self.ids[atom_id]]
            sites = [self.sites[atom_id]]
        else:
            ids = self.ids
            sites = self.sites

        lines = [
            'ITEM: TIMESTEP',
            timestep,
            'ITEM: NUMBER OF ATOMS',
            len(ids),
            f'ITEM: BOX BOUNDS {self.dump_cache[1]}'
        ]

        for upper_bound in self.bounds:
            lines.append(f'0.0 {upper_bound}')

        lines.append(f'ITEM: ATOMS id type {self.dump_cache[0]}{" " + additional_labels if additional_labels else ""}')

        for i, site in enumerate(sites):
            formatted_site = ' '.join(map(str, site))
            additional_txt = ''
            if additional_coords is not None:
                additional_txt = ' ' + ' '.join(map(str, additional_coords[i]))
            lines.append(
                f'{i + 1:.0f} 1 {formatted_site}{additional_txt}')

        for line in lines:
            print(line, file=file, flush=True)

    def get_distance(self, site_1: np.ndarray, site_2: np.ndarray, periodic: bool = True) -> float:
        """
        Get the distance between two sites
        :param site_1: First site
        :param site_2: Second site
        :param periodic: If true, periodic boundary conditions are applied
        :return: Distance between the two sites
        """
        if periodic:
            return np.linalg.norm(np.mod(site_1 - site_2 + self.bounds / 2, self.bounds) - self.bounds / 2)
        return np.linalg.norm(site_1 - site_2)

    def crossed_boundary(self, site_1: int, site_2: int) -> np.ndarray:
        """
        Determine which boundaries a line from site_1 to site_2 crosses in a periodic system.

        :param site_1: ID of the first site.
        :param site_2: ID of the second site.
        :return: An int array representing the direction of the boundary crossing. 0 means no crossing, 1 means
        crossing from positive to negative, and -1 means crossing from negative to positive.
        """
        crossing = np.abs(self.sites[site_1] - self.sites[site_2]) > self.bounds / 2
        return (np.sign(self.sites[site_1] - self.sites[site_2]) * crossing).astype(int)

    @abstractmethod
    def get_lat_info(self) -> LatInfo:
        pass

    def _calc_dump_cache(self):

        cartesian_labels = ['x', 'y', 'z']

        if len(self.dimensions) <= 3:
            coord_names = cartesian_labels[:len(self.dimensions)]
        else:
            coord_names = [f'x{i:.0f}' for i in range(len(self.dimensions))]

        used_coord_names = ' '.join(coord_names)
        bounds = " ".join(len(self.dimensions) * ["pp"])
        self.dump_cache = (used_coord_names, bounds)

    @abstractmethod
    def get_adjacency_matrix(self, epsilon=0.1, strict=False) -> np.ndarray:
        pass


@dataclass
class OrthogonalLattice(Lattice):
    """
    Lattice class for orthogonal lattices
    """

    class LatticeType(Enum):
        SC = 0
        FCC = 1
        BCC = 2

    lattice_type: LatticeType

    def __init__(self, lattice_type: Union[LatticeType, str, int, None] = None,
                 dimensions: Union[ArrayLike, None] = None,
                 lattice_vector: Union[ArrayLike, None] = None,
                 lat_info: Union[LatInfo, None] = None):
        """
        Initialize an orthogonal lattice
        :param lattice_type: Lattice type
        :param dimensions: Dimensions of the lattice
        :param lattice_vector: vector between lattice sites
        :param lat_info: LatInfo object containing other information about the lattice (for multiprocessing)
        """
        super().__init__(dimensions=None, lattice_vector=None, sites=None, ids=None, bounds=None, num_sites=None,
                         coordination_number=None, adjacency_matrix=None, dump_cache=None, shared_memory=None,
                         original=None)
        if lat_info is None and (lattice_type is None or dimensions is None or lattice_vector is None):
            raise ValueError("lattice_type, dimensions, and lattice_vector must be specified")
        if lat_info is not None:
            self._init_from_lat_info(lat_info)
            return
        self.original = True
        if isinstance(lattice_type, str):
            try:
                lattice_type = getattr(self.LatticeType, lattice_type.upper())
            except AttributeError:
                warn("Invalid lattice type")
                raise
        dimensions_num = len(dimensions)

        if len(dimensions) != len(lattice_vector):
            raise ValueError("Dimensions and lattice_vector must have the same length")
        if len(dimensions) > 3 and lattice_type != OrthogonalLattice.LatticeType.SC:
            raise ValueError("Only a maximum of 3 dimensions are supported for fcc and bcc lattices")
        # Pad dimensions and lattice_vector to 3 dimensions
        if len(dimensions) == 1:
            dimensions = np.array([dimensions[0], 1, 1])
            lattice_vector = np.ones(3) * lattice_vector[0]
        elif len(dimensions) == 2:
            dimensions = np.array([dimensions[0], dimensions[1], 1])
            lattice_vector = np.array([lattice_vector[0], lattice_vector[1], 1])
        # Convert to numpy arrays
        lattice_vector = np.array(lattice_vector)
        dimensions = np.array(dimensions)
        # Create shared memory blocks for the lattice vector and dimensions
        shm_lat_vec = SharedMemory(create=True, size=lattice_vector.nbytes)
        shm_dim = SharedMemory(create=True, size=dimensions.nbytes)
        # Copy the lattice vector and dimensions into the shared memory blocks
        buffer = np.ndarray(lattice_vector.shape, dtype=lattice_vector.dtype, buffer=shm_lat_vec.buf)
        buffer[:] = lattice_vector[:]
        buffer = np.ndarray(dimensions.shape, dtype=dimensions.dtype, buffer=shm_dim.buf)
        buffer[:] = dimensions[:]
        # Move the shared memory blocks into the class
        self.lattice_vector = np.ndarray(lattice_vector.shape, dtype=lattice_vector.dtype, buffer=shm_lat_vec.buf)
        self.dimensions = np.ndarray(dimensions.shape, dtype=dimensions.dtype, buffer=shm_dim.buf)
        del lattice_vector
        del dimensions
        self.lattice_type = lattice_type
        curr_id = 0
        num_sites = np.prod(self.dimensions) * len(self.atomic_basis)
        sites = np.zeros((num_sites, len(self.dimensions)))
        ids = np.zeros(num_sites)
        shm_sites = SharedMemory(create=True, size=sites.nbytes)
        shm_ids = SharedMemory(create=True, size=ids.nbytes)
        buffer = np.ndarray(sites.shape, dtype=sites.dtype, buffer=shm_sites.buf)
        buffer[:] = sites[:]
        buffer = np.ndarray(ids.shape, dtype=ids.dtype, buffer=shm_ids.buf)
        buffer[:] = ids[:]
        self.sites = np.ndarray(sites.shape, dtype=sites.dtype, buffer=shm_sites.buf)
        self.ids = np.ndarray(ids.shape, dtype=ids.dtype, buffer=shm_ids.buf)
        del sites
        del ids

        for coord in product(*[range(i) for i in self.dimensions]):
            lattice_vector = np.array(coord) * self.lattice_vector
            for basis_site in self.atomic_basis:
                site = lattice_vector + basis_site * self.lattice_vector
                self.sites[curr_id] = site
                self.ids[curr_id] = curr_id
                curr_id += 1
        bounds = np.max(self.sites, axis=0)
        shm_bounds = SharedMemory(create=True, size=bounds.nbytes)
        buffer = np.ndarray(bounds.shape, dtype=bounds.dtype, buffer=shm_bounds.buf)
        buffer[:] = bounds[:]
        self.bounds = np.ndarray(bounds.shape, dtype=bounds.dtype, buffer=shm_bounds.buf)
        del bounds
        self.shared_memory = LatSharedMemory(lattice_vector=shm_lat_vec, dimensions=shm_dim, sites=shm_sites,
                                             ids=shm_ids, bounds=shm_bounds, adjacency_matrix=None)
        del buffer
        if (self.lattice_type == OrthogonalLattice.LatticeType.FCC or
                self.lattice_type == OrthogonalLattice.LatticeType.BCC):
            self.bounds += 0.5 * self.lattice_vector
        elif self.lattice_type == OrthogonalLattice.LatticeType.SC:
            self.bounds += self.lattice_vector
        self.num_sites = len(self.sites)
        self.adjacency_matrix = None
        self._calc_dump_cache()

        if self.lattice_type == OrthogonalLattice.LatticeType.SC:
            self.coordination_number = 2 * dimensions_num
        elif self.lattice_type == OrthogonalLattice.LatticeType.FCC:
            self.coordination_number = 12
        elif self.lattice_type == OrthogonalLattice.LatticeType.BCC:
            self.coordination_number = 8
        else:
            raise ValueError("Invalid lattice type")

    def __del__(self):
        if self.original:
            for shm in self.shared_memory.all:
                if shm is None:
                    continue
                shm.close()
                shm.unlink()
            return
        for shm in self.shared_memory.all:
            shm.close()

    @property
    def atomic_basis(self) -> np.ndarray:
        if self.lattice_type == OrthogonalLattice.LatticeType.SC:
            return np.array([np.zeros(len(self.dimensions))])
        if self.lattice_type == OrthogonalLattice.LatticeType.FCC:
            return np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        if self.lattice_type == OrthogonalLattice.LatticeType.BCC:
            return np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    def _init_from_lat_info(self, lat_info: LatInfo):
        self.original = False
        self.num_sites = lat_info.num_sites
        if lat_info.lat_type is not None:
            self.lattice_type = OrthogonalLattice.LatticeType(lat_info.lat_type)
        self.shared_memory = LatSharedMemory(
            lattice_vector=SharedMemory(name=lat_info.lattice_vector),
            dimensions=SharedMemory(name=lat_info.dimensions),
            sites=SharedMemory(name=lat_info.sites),
            ids=SharedMemory(name=lat_info.ids),
            bounds=SharedMemory(name=lat_info.bounds),
            adjacency_matrix=SharedMemory(name=lat_info.adjacency_matrix))
        for i, name in enumerate(self.shared_memory.all_actual_names):
            if getattr(self, name) is not None:
                continue
            setattr(self, name, np.ndarray(lat_info.all_shapes[i], dtype=lat_info.all_types[i],
                                           buffer=self.shared_memory.all[i].buf))
        self.coordination_number = np.sum(self.adjacency_matrix, axis=0)[0]
        self._calc_dump_cache()

    @overrides
    def get_lat_info(self) -> LatInfo:
        if self.adjacency_matrix is None:
            warn("Adjacency matrix is not calculated yet.. This may take a while")
            self.get_adjacency_matrix()
        return LatInfo(lattice_vector=self.shared_memory.lattice_vector.name,
                       dimensions=self.shared_memory.dimensions.name,
                       sites=self.shared_memory.sites.name,
                       ids=self.shared_memory.ids.name,
                       bounds=self.shared_memory.bounds.name,
                       adjacency_matrix=self.shared_memory.adjacency_matrix.name,
                       lat_vector_type=self.lattice_vector.dtype,
                       dimensions_type=self.dimensions.dtype,
                       sites_type=self.sites.dtype,
                       ids_type=self.ids.dtype,
                       bounds_type=self.bounds.dtype,
                       adjacency_matrix_type=self.adjacency_matrix.dtype,
                       lat_shape=self.lattice_vector.shape,
                       dimensions_shape=self.dimensions.shape,
                       sites_shape=self.sites.shape,
                       ids_shape=self.ids.shape,
                       bounds_shape=self.bounds.shape,
                       adjacency_matrix_shape=self.adjacency_matrix.shape,
                       num_sites=self.num_sites,
                       lat_type=self.lattice_type.value)

    def del_adjacency_matrix(self):
        if not self.original:
            raise ValueError("Cannot delete adjacency matrix of shared memory lattice")
        if self.adjacency_matrix is None:
            return
        self.adjacency_matrix = None
        self.shared_memory.adjacency_matrix.close()
        self.shared_memory.adjacency_matrix.unlink()
        self.shared_memory.adjacency_matrix = None

    @overrides
    def get_adjacency_matrix(self, epsilon=0.1, strict=False) -> np.ndarray:
        """
        Get the adjacency matrix of the lattice
        This is an expensive operation, so the result is cached. To recalculate the adjacency matrix, use the
        del_adjacency_matrix method
        :param epsilon: Epsilon for the cutoff
        :param strict: If true, an error is raised if a site has an invalid coordination number
        :return: Adjacency matrix
        """
        if self.adjacency_matrix is not None:
            return self.adjacency_matrix
        if not self.original:
            raise ValueError("Cannot calculate adjacency matrix of shared memory lattice")
        cutoff = 1 + epsilon
        if self.lattice_type == OrthogonalLattice.LatticeType.SC:
            cutoff *= self.lattice_vector[0]
        elif self.lattice_type == OrthogonalLattice.LatticeType.FCC:
            cutoff *= self.lattice_vector[0] * 1 / 2 * (1 / np.sqrt(2) + 1)
        else:
            # BCC
            cutoff *= self.lattice_vector[0] * (np.sqrt(3)) / 2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sites_tensor = torch.tensor(self.sites, device=device)

        # Calculate the adjacency matrix

        dimensions_tensor = torch.tensor(self.dimensions, device=device)
        lattice_parameter = self.lattice_vector[0]

        cell_matrix = lattice_parameter * torch.diag(dimensions_tensor)
        del dimensions_tensor, lattice_parameter
        cell_matrix = cell_matrix.double()
        inverted_cell_matrix = torch.linalg.inv(cell_matrix)

        physical_differences = sites_tensor[:, None] - sites_tensor

        fractional_differences = torch.einsum('km,ijm->ijk', inverted_cell_matrix, physical_differences)
        del inverted_cell_matrix
        images = torch.einsum('km,ijm->ijk', cell_matrix, torch.round(fractional_differences))
        del cell_matrix, fractional_differences
        minimum_image_differences = physical_differences - images
        del physical_differences, images

        minimum_image_distances = torch.linalg.norm(minimum_image_differences, dim=2)
        adjacency_matrix = (0 < minimum_image_distances) & (minimum_image_distances <= cutoff)
        del minimum_image_distances, minimum_image_differences

        adjacency_matrix = adjacency_matrix.cpu().numpy().astype(np.uint8)

        # Create a shared memory block for the adjacency matrix
        adjacency_matrix_shm = SharedMemory(create=True, size=adjacency_matrix.nbytes)
        self.adjacency_matrix = np.ndarray(adjacency_matrix.shape, dtype=np.uint8, buffer=adjacency_matrix_shm.buf)
        self.adjacency_matrix[:] = adjacency_matrix[:]
        del adjacency_matrix
        self.shared_memory.adjacency_matrix = adjacency_matrix_shm

        # check if all sites have the correct coordination number
        for i, row in enumerate(self.adjacency_matrix):
            if np.sum(row) == self.coordination_number:
                continue
            if strict:
                raise ValueError("Invalid lattice")
            warn(f"Site {i} has an invalid coordination number")

        return self.adjacency_matrix
