from itertools import combinations
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
from typing import Tuple, Callable, Union, Any
import numpy as np


@dataclass
class ECInfo:
    """
    Class for storing information about an energy container
    """
    energies_name: str
    energies_type: np.dtype
    energies_dimensions: Tuple[int, ...]


@dataclass
class EnergyContainer:
    """
    Class for storing energies of a system
    """

    energy_array: np.ndarray
    energy_shm: SharedMemory
    original: bool

    def __init__(self, ec_info: Union[ECInfo, None] = None, energy_array: Union[np.ndarray, None] = None,
                 energy_shm: Union[SharedMemory, None] = None, original: Union[bool, None] = None):
        """
        Initialize the EnergyContainer
        :param ec_info: EnergyContainer information
        :param energy_array: energy array
        :param energy_shm: energy shared memory
        :param original: whether this is the original EnergyContainer
        """
        if ec_info is not None:
            self._init_from_info(ec_info)
            return
        if energy_array is None:
            raise ValueError('Must provide an energy array or ECInfo')
        if energy_shm is None:
            raise ValueError('Must provide an energy shared memory or ECInfo')
        if original is None:
            raise ValueError('Must provide whether this is the original EnergyContainer')
        if not isinstance(energy_array, np.ndarray):
            raise TypeError('Energy array must be a numpy array')
        if not isinstance(energy_shm, SharedMemory):
            raise TypeError('Energy shared memory must be a SharedMemory object')
        self.energy_array = energy_array
        self.energy_shm = energy_shm
        self.original = original

    def __del__(self):
        """
        Delete the EnergyContainer
        """
        self.energy_shm.close()
        if self.original:
            self.energy_shm.unlink()

    def _init_from_info(self, ec_info: ECInfo):
        self.original = False
        self.energy_shm = SharedMemory(name=ec_info.energies_name)
        self.energy_array = np.ndarray(ec_info.energies_dimensions, dtype=ec_info.energies_type,
                                       buffer=self.energy_shm.buf)

    def get_ec_info(self) -> ECInfo:
        """
        Get the ECInfo for this EnergyContainer
        :return: ECInfo
        """
        return ECInfo(energies_name=self.energy_shm.name, energies_type=self.energy_array.dtype,
                      energies_dimensions=self.energy_array.shape)

    @staticmethod
    def _call_generator_function(func, args, kwargs):
        if args is None:
            if kwargs is None:
                return func()
            return func(**kwargs)
        if kwargs is None:
            return func(*args)
        return func(*args, **kwargs)


@dataclass
class InitialEnergies(EnergyContainer):
    """
    Class for storing initial energies of a system
    """

    def __init__(self, num_sites: Union[int, None] = None, generator_function: Union[Callable[..., Union[float, np.ndarray]], None] = None,
                 function_args: Union[Any, None] = None, function_kwargs: Union[Any, None] = None, ec_info: Union[ECInfo, None] = None):
        """
        Initialize the EnergyContainer
        :param num_sites: number of sites in the system
        :param generator_function: function to generate the energy array (if it returns a numpy array,
        it needs to return a 1D array of length num_sites)
        :param function_args: function arguments
        :param function_kwargs: function keyword arguments
        :param ec_info: EnergyContainer information
        """
        if ec_info is not None:
            super().__init__(ec_info=ec_info)
            return
        if generator_function is None:
            raise ValueError('Must provide a generator function or ECInfo')
        if function_kwargs is None:
            function_kwargs = {}
        function_test = self._call_generator_function(generator_function, function_args, function_kwargs)
        if not isinstance(function_test, float) and not isinstance(function_test, np.ndarray):
            raise TypeError('Generator function must return a float or numpy array')
        if isinstance(function_test, float) and num_sites is None:
            raise ValueError('Must provide the number of sites in the system')
        if isinstance(function_test, float):
            energy_array = np.array([self._call_generator_function(generator_function, function_args, function_kwargs) for _ in range(num_sites)])
        else:
            energy_array = self._call_generator_function(generator_function, function_args, function_kwargs)
        if len(energy_array.shape) != 1 or energy_array.shape[0] != num_sites:
            raise ValueError('Generator function must return a 1D array of length num_sites')
        temp_energy_shm = SharedMemory(create=True, size=energy_array.nbytes)
        temp_energy_array = np.ndarray(energy_array.shape, dtype=energy_array.dtype, buffer=temp_energy_shm.buf)
        temp_energy_array[:] = energy_array[:]
        del energy_array
        temp_original = True
        super().__init__(energy_array=temp_energy_array, energy_shm=temp_energy_shm, original=temp_original)


@dataclass
class SaddleEnergies(EnergyContainer):
    """
    Class for storing saddle energies of a system
    """

    def __init__(self, adjacency_matrix: Union[np.ndarray, None], generator_function: Union[Callable[..., Union[float, np.ndarray]], None],
                 function_args: Union[Any, None] = None, function_kwargs: Union[Any, None] = None,
                 init_energies: Union[InitialEnergies, None] = None, ec_info: Union[ECInfo, None] = None):
        """
        Initialize the EnergyContainer
        :param adjacency_matrix: adjacency matrix for the system
        :param generator_function: function to generate the energy array. If it returns a numpy array, it needs to return
        a 2D array of shape (num_sites, num_sites) that is symmetric along the diagonal (not strictly enforced). If a numpy
        array is returned it won't check that the saddle energies are higher than the initial energies
        :param function_args: function arguments
        :param function_kwargs: function keyword arguments
        :param init_energies: InitialEnergies object
        :param ec_info: EnergyContainer information
        """
        if ec_info is not None:
            super().__init__(ec_info=ec_info)
            return
        if generator_function is None:
            raise ValueError('Must provide a generator function or ECInfo')
        if adjacency_matrix is None:
            raise ValueError('Must provide an adjacency matrix or ECInfo')
        function_test = self._call_generator_function(generator_function, function_args, function_kwargs)
        if not isinstance(function_test, float) and not isinstance(function_test, np.ndarray):
            raise TypeError('Generator function must return a float')
        if isinstance(function_test, float) and init_energies is None:
            raise ValueError('Must provide an InitialEnergies object')
        if isinstance(function_test, float):
            energy_array = np.zeros(adjacency_matrix.shape)
            for i, j in combinations(range(adjacency_matrix.shape[0]), 2):
                if not adjacency_matrix[i, j]:
                    continue
                energy = -np.inf
                while energy < init_energies.energy_array[i] or energy < init_energies.energy_array[j]:
                    energy = self._call_generator_function(generator_function, function_args, function_kwargs)
                energy_array[i, j] = energy
            energy_array += energy_array.T
        else:
            energy_array = self._call_generator_function(generator_function, function_args, function_kwargs)
            if energy_array.shape != adjacency_matrix.shape:
                raise ValueError('Generator function must return a 2D array of shape (num_sites, num_sites)')
        temp_energy_shm = SharedMemory(create=True, size=energy_array.nbytes)
        temp_energy_array = np.ndarray(energy_array.shape, dtype=energy_array.dtype, buffer=temp_energy_shm.buf)
        temp_energy_array[:] = energy_array[:]
        del energy_array
        temp_original = True
        super().__init__(energy_array=temp_energy_array, energy_shm=temp_energy_shm, original=temp_original)
