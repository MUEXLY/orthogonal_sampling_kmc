import lzma
import numpy as np
import time
from multiprocessing import Pool, Manager, Process
from dataclasses import dataclass
from typing import Tuple, IO, Union, List
from kmc.lattice import Lattice, LatInfo, OrthogonalLattice
from bisect import bisect_left
from random import seed, randint
from uuid import uuid4
from kmc.energy_container import ECInfo, EnergyContainer

BOLTZMANN_CONSTANT = 8.617e-5 # eV/K


@dataclass
class KMCInfo:
    """
    Class for storing information about a kinetic Monte Carlo simulation
    """
    lat_info: LatInfo
    initial_info: ECInfo
    saddle_info: ECInfo
    temp: float


@dataclass
class KineticMonteCarlo:
    """
    Class for performing kinetic Monte Carlo simulations
    """

    crystal: Lattice
    initial_energies_obj: EnergyContainer
    saddle_energies_obj: EnergyContainer
    initial_energies: np.ndarray
    saddle_energies: np.ndarray
    boundary_crossings: np.ndarray
    dump_cache: str
    rates: np.ndarray
    transition_states: np.ndarray
    temp: float
    vacancy_id: int
    num_sites: int
    curr_step: int
    time: float
    rate_prefactor: float = 1e+13
    testing: bool = False

    def __init__(self, crystal: Union[Lattice, None] = None, initial_energies: Union[EnergyContainer, None] = None,
                 saddle_energies: Union[EnergyContainer, None] = None, temp: float = 300, rand_seed: int = None,
                 kmc_info: Union[KMCInfo, None] = None):
        """
        Initialize a kinetic Monte Carlo simulation
        :param crystal: Crystal to perform the simulation on
        :param initial_energies: Initial energies of the crystal
        :param saddle_energies: Saddle energies of the crystal
        :param temp: Temperature of the simulation
        :param kmc_info: Information about the simulation
        """
        if kmc_info is not None:
            self.__init_from_info(kmc_info)
            return
        if crystal is None or initial_energies is None or saddle_energies is None:
            raise ValueError('Must provide a crystal, initial energies, and saddle energies or KMCInfo')
        self.crystal = crystal
        self.boundary_crossings = np.zeros(self.crystal.bounds.shape[0], dtype=int)
        self.num_sites = self.crystal.num_sites
        self.initial_energies_obj = initial_energies
        self.saddle_energies_obj = saddle_energies
        self.initial_energies = self.initial_energies_obj.energy_array
        self.saddle_energies = self.saddle_energies_obj.energy_array
        self.curr_step = 0
        self.time = 0.0
        self.temp = temp * BOLTZMANN_CONSTANT
        # Pick a random atom as our vacancy
        # Use builtin random generator with seed provided from UUID4
        if rand_seed is not None:
            seed(rand_seed)
        else:
            seed(int(uuid4()))
        self.vacancy_id = randint(0, self.num_sites - 1)
        self.rates = np.zeros(self.crystal.coordination_number)
        self.transition_states = np.zeros(self.rates.shape, dtype=int)
        self._build_dump_cache()

    def _build_dump_cache(self):
        """
        Build the cache for dumping the lattice
        """
        cartesian_labels = ['x', 'y', 'z']

        if len(self.crystal.dimensions) <= 3:
            coord_names = cartesian_labels[:len(self.crystal.dimensions)]
        else:
            coord_names = [f'x{i:.0f}' for i in range(len(self.crystal.dimensions))]

        coord_names = [f'i{c}' for c in coord_names]
        self.dump_cache = ' '.join(coord_names)

    def __init_from_info(self, kmc_info: KMCInfo):
        self.original = False
        self.crystal = OrthogonalLattice(lat_info=kmc_info.lat_info)
        self.num_sites = self.crystal.num_sites
        self.initial_energies_obj = EnergyContainer(ec_info=kmc_info.initial_info)
        self.saddle_energies_obj = EnergyContainer(ec_info=kmc_info.saddle_info)
        self.initial_energies = self.initial_energies_obj.energy_array
        self.saddle_energies = self.saddle_energies_obj.energy_array
        self.temp = kmc_info.temp
        seed(int(uuid4()))
        self.vacancy_id = randint(0, self.num_sites - 1)
        self.curr_step = 0
        self.time = 0.0
        self.rates = np.zeros(self.crystal.coordination_number)
        self.transition_states = np.zeros(self.rates.shape, dtype=int)
        self.boundary_crossings = np.zeros(self.crystal.bounds.shape[0], dtype=int)
        self._build_dump_cache()

    def get_kmc_info(self) -> KMCInfo:
        """
        Get the information about this simulation
        """
        return KMCInfo(
            lat_info=self.crystal.get_lat_info(),
            initial_info=self.initial_energies_obj.get_ec_info(),
            saddle_info=self.saddle_energies_obj.get_ec_info(),
            temp=self.temp
        )

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix for the crystal
        """
        return self.crystal.get_adjacency_matrix()

    def dump(self, file: IO, atom_id: int = None):
        if atom_id is None:
            self.crystal.dump(file, timestep=(self.curr_step, self.time), suppress_warning=True)
        else:
            self.crystal.dump(file, timestep=(self.curr_step, self.time), atom_id=atom_id, suppress_warning=True,
                              additional_labels=self.dump_cache, additional_coords=[self.boundary_crossings])

    def step(self):
        """
        Perform a step in the simulation
        """

        self.rates.fill(0)
        self.transition_states.fill(0)

        adjacency_row = self.adjacency_matrix[self.vacancy_id, :]
        saddle_row = self.saddle_energies[self.vacancy_id, :]

        transition_index = 0
        for i, neighbor in enumerate(adjacency_row):
            if not neighbor:
                continue
            self.transition_states[transition_index] = i
            transition_index += 1

        saddle_points = saddle_row[self.transition_states]
        initial_points = self.initial_energies[self.transition_states]

        self.rates = self.rate_prefactor * np.exp(-(saddle_points - initial_points) / self.temp)
        # Rates are generated in a deterministic order, so we need to shuffle them
        np.random.shuffle(self.rates)
        sorted_indices = self.rates.argsort()
        sorted_rates, sorted_states = self.rates[sorted_indices], self.transition_states[sorted_indices]
        cumulative_function = np.cumsum(sorted_rates)
        total_rate = cumulative_function[-1]

        random_draw = 1.0 - np.random.uniform(low=0, high=1)
        event_index = bisect_left(cumulative_function, random_draw * total_rate)

        second_draw = 1.0 - np.random.uniform(low=0, high=1)
        self.time += np.log(1.0 / second_draw) * 1.0 / total_rate
        self.curr_step += 1
        previous_vacancy = self.vacancy_id
        self.vacancy_id = int(sorted_states[event_index])
        assert self.vacancy_id is not None and self.vacancy_id != previous_vacancy

        # We have to update the boundary crossings
        self.boundary_crossings += self.crystal.crossed_boundary(previous_vacancy, self.vacancy_id)

    def run(self, steps: int, vacancy_dump_file: IO, whole_lattice_dump_file: IO = None, dump_lat_every: int = None,
            verbose: bool = True, async_queue=None) -> None:
        """
        Run the simulation for a number of steps
        :param verbose: Print status updates
        :param steps: Number of steps to run the simulation for
        :param vacancy_dump_file: File to dump vacancy trajectory to
        :param whole_lattice_dump_file: File to dump whole lattice trajectory to
        :param dump_lat_every: Dump the whole lattice every this many steps
        :param async_queue: Queue to put status updates in when using run_many
        """
        assert self.vacancy_id is not None
        start_time = time.time()
        if whole_lattice_dump_file is None and vacancy_dump_file is None:
            raise ValueError('Must provide a file to dump the either the whole lattice or vacancy trajectory to')
        for i in range(steps):
            if dump_lat_every and i % dump_lat_every == 0 and vacancy_dump_file:
                self.dump(vacancy_dump_file, atom_id=self.vacancy_id)
            if dump_lat_every and i % dump_lat_every == 0 and whole_lattice_dump_file:
                self.dump(whole_lattice_dump_file)
            self.step()
            # Print status every 100 steps
            if verbose and i % 100 == 0:
                time_elapsed = time.time() - start_time
                time_total = time_elapsed / (i + 1) * steps
                print(f'Completed step {i} of {steps}|{time_elapsed:.2f}s / {time_total:.2f}s', end='\r')
            if async_queue is not None and i % 10_000 == 0:
                async_queue.put(i)
        if verbose:
            print(
                f'Completed step {steps} of {steps}|{time.time() - start_time:.2f}s / {time.time() - start_time:.2f}s')
        if async_queue is not None:
            async_queue.put(-1)  # Signal that the simulation is done

    @staticmethod
    def _single_thread_run(kmc_info: KMCInfo, steps: int, vacancy_dump_file_name: str, file_encoding: str,
                           dump_every_n: int, queue) -> None:
        kmc = KineticMonteCarlo(kmc_info=kmc_info)
        if vacancy_dump_file_name.endswith('.xz'):
            with lzma.open(vacancy_dump_file_name, 'wt', encoding=file_encoding) as vacancy_dump_file:
                kmc.run(steps, vacancy_dump_file, verbose=False, dump_lat_every=dump_every_n, async_queue=queue)
            return
        with open(vacancy_dump_file_name, 'w', encoding=file_encoding) as vacancy_dump_file:
            kmc.run(steps, vacancy_dump_file, verbose=False, dump_lat_every=dump_every_n, async_queue=queue)

    @staticmethod
    def _printer_thread(runs, start_time, total_steps, queue):
        if queue is None:
            return
        threads_done = 0
        steps_completed = 0
        print(f'Running {runs} simulations for {total_steps} steps...')
        while threads_done < runs:
            status = queue.get()
            if status == -1:
                threads_done += 1
                continue
            steps_completed += 10_000
            time_elapsed = time.time() - start_time
            time_total = time_elapsed / steps_completed * total_steps
            print(f'Completed step {steps_completed} of {total_steps}|{time_elapsed:.2f}s / {time_total:.2f}s',
                  end='\r')
        print(
            f'Finished running {runs} simulations for {total_steps} steps! Total time: {time.time() - start_time:.2f}s')

    def run_many(self, steps: int, vacancy_dump_files: Union[Tuple[str, ...], List[str]], file_encoding: str = 'utf_8',
                 dump_every_n: int = 1, verbose: bool = True) -> None:
        """
        Run the simulation for a number of steps, but dump to multiple files
        :param steps: number of steps to run the simulation for
        :param vacancy_dump_files: files to dump vacancy trajectory to
        :param file_encoding: encoding of the dump files
        :param dump_every_n: dump to every n files
        :param verbose: print status updates
        :return: None
        """
        runs = len(vacancy_dump_files)
        total_steps = steps * runs
        start_time = time.time()
        with Manager() as manager:
            queue = manager.Queue() if verbose else None
            printer_thread = Process(target=self._printer_thread, args=(runs, start_time, total_steps, queue))
            printer_thread.start()
            args = [(self.get_kmc_info(), steps, vacancy_dump_files[i], file_encoding, dump_every_n, queue) for i in
                    range(runs)]
            with Pool() as pool:
                pool.starmap(self._single_thread_run, args)
            printer_thread.join()
