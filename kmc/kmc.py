import csv
import random
import pandas as pd
from random import shuffle, randint
import uuid

from scipy.constants import physical_constants
from tqdm import tqdm

from analysis import KMCv2Analyzer
from .lattice import Lattice, OrthogonalLattice
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Iterable, Any
from pathlib import Path
import sqlite3 as sql
import numpy as np
import scipy.stats as stats
import multiprocessing as mp


def get_params_from_csv(csv_file: str, file_encoding: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Get the parameters from a CSV file.
    :param csv_file: CSV file to read from.
    :param file_encoding: Encoding of the CSV file.
    :return: Two Tuples containing the initial mean and sd and the saddle mean and sd.
    """
    with open(csv_file, 'r', encoding=file_encoding) as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    init_energies = []
    saddle_energies = []
    for i in range(len(rows)):
        if i == 0:
            continue
        init_energies.append(float(rows[i][2]))
        saddle_energies.append(float(rows[i][3]))
    init_energies = np.array(init_energies)
    saddle_energies = np.array(saddle_energies)

    def reject_outliers(data, m=100.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else np.zeros(len(d))
        return data[s < m]

    init_energies = reject_outliers(init_energies)
    saddle_energies = reject_outliers(saddle_energies)

    init_params = (np.mean(init_energies), np.std(init_energies))
    saddle_params = (np.mean(saddle_energies), np.std(saddle_energies))
    return init_params, saddle_params

class Sampler(ABC):

    @staticmethod
    def _index_sym(l:List[int] | Tuple[int,...]) -> Tuple[int,...]:
        return tuple(sorted(l))

    @abstractmethod
    def get_rates(self, idx:int) -> Tuple[np.ndarray[2], float]:
        pass

    @abstractmethod
    def get_random(self) -> int:
        pass

    @abstractmethod
    def write_to_db(self, con:sql.Connection) -> None:
        pass


# noinspection PyTypeChecker
class DualSampler(Sampler):
    init_energies: np.ndarray
    saddle_energies: Dict[Tuple[int,int], float]
    neighbors: Dict[int, List[int]]
    rates: Dict[int, List[Tuple[int, float]]]
    rates_sum: Dict[int, float]

    temp: float
    init_params: Tuple[float, float]
    saddle_params: Tuple[float, float]
    prefactor: float

    def __init__(self, lattice:Lattice, init_params:(float, float), saddle_params:(float, float), temp:float, prefactor:float = 1e13):
        self.init_energies = np.random.normal(init_params[0], init_params[1], len(lattice.get_adjacency_matrix()))
        self.saddle_energies = {}
        self.neighbors = {}
        self.temp = temp

        temp_energy = temp * physical_constants['Boltzmann constant in eV/K'][0]

        self.init_params = init_params
        self.saddle_params = saddle_params
        self.prefactor = prefactor
        for i, row in enumerate(lattice.get_adjacency_matrix()):
            self.neighbors[i] = []
            for j, val in enumerate(row):
                if val == 1:
                    self.saddle_energies[Sampler._index_sym((i, j))] = np.random.normal(saddle_params[0], saddle_params[1])
                    self.neighbors[i].append(j)

        for i in range(len(self.init_energies)):
            for n in self.neighbors[i]:
                if self.saddle_energies[Sampler._index_sym((i,n))] < self.init_energies[i]:
                    self.saddle_energies[Sampler._index_sym((i,n))] = float(self.init_energies[i])
        self.rates = {}
        self.rates_sum = {}
        for i in range(len(self.init_energies)):
            rates = []
            rates_sum = 0.0
            for n in self.neighbors[i]:
                rate = prefactor * np.exp(-(self.saddle_energies[Sampler._index_sym((i,n))] - self.init_energies[i]) / temp_energy)
                rates.append((n, rate))
                rates_sum += rate
            self.rates[i] = rates
            self.rates_sum[i] = rates_sum

    def get_rates(self, idx:int) -> Tuple[List[Tuple[int, float]], float]:
        shuffle(self.rates[idx])
        return self.rates[idx], self.rates_sum[idx]

    def get_random(self) -> int:
        return randint(0, len(self.init_energies)-1)

    # noinspection SqlWithoutWhere
    def write_to_db(self, con:sql.Connection) -> None:
        con.execute('CREATE TABLE IF NOT EXISTS dual_sampler_params (key TEXT PRIMARY KEY, value TEXT)')
        con.execute('CREATE TABLE IF NOT EXISTS dual_sampler_init_energies (id INT PRIMARY KEY, energy REAL)')
        con.execute('CREATE TABLE IF NOT EXISTS dual_sampler_saddle_energies (i INT, j INT, energy REAL)')
        con.execute('CREATE TABLE IF NOT EXISTS dual_sampler_rates (id INT, neighbor INT, rate REAL)')
        con.execute('CREATE TABLE IF NOT EXISTS dual_sampler_barriers (id INT, neighbor INT, energy REAL)')

        con.execute('DELETE FROM dual_sampler_params')
        con.execute('DELETE FROM dual_sampler_init_energies')
        con.execute('DELETE FROM dual_sampler_saddle_energies')
        con.execute('DELETE FROM dual_sampler_rates')
        con.execute('DELETE FROM dual_sampler_barriers')

        con.execute('INSERT INTO dual_sampler_params (key, value) VALUES ("temp", ?)', (self.temp,))
        con.execute('INSERT INTO dual_sampler_params (key, value) VALUES ("prefactor", ?)', (self.prefactor,))
        con.execute('INSERT INTO dual_sampler_params (key, value) VALUES ("init_mean", ?)', (str(self.init_params[0]),))
        con.execute('INSERT INTO dual_sampler_params (key, value) VALUES ("init_sd", ?)', (str(self.init_params[1]),))
        con.execute('INSERT INTO dual_sampler_params (key, value) VALUES ("saddle_mean", ?)', (str(self.saddle_params[0]),))
        con.execute('INSERT INTO dual_sampler_params (key, value) VALUES ("saddle_sd", ?)', (str(self.saddle_params[1]),))
        con.executemany('INSERT INTO dual_sampler_init_energies (id, energy) VALUES (?, ?)', [(i, e) for i, e in enumerate(self.init_energies)])
        for pair_id in self.saddle_energies:
            con.execute('INSERT INTO dual_sampler_saddle_energies (i,j , energy) VALUES (?, ?, ?)', (pair_id[0], pair_id[1], self.saddle_energies[pair_id]))
        for idx in self.rates:
            for neighbor, rate in self.rates[idx]:
                con.execute('INSERT INTO dual_sampler_rates (id, neighbor, rate) VALUES (?, ?, ?)', (idx, neighbor, rate))
        for idx in self.neighbors:
            for neighbor in self.neighbors[idx]:
                energy = self.saddle_energies[Sampler._index_sym((idx, neighbor))] - self.init_energies[idx]
                con.execute('INSERT INTO dual_sampler_barriers (id, neighbor, energy) VALUES (?, ?, ?)', (idx, neighbor, energy))
        con.commit()


class KMCv2:
    time: float = 0
    commit_every: int
    idx: int
    sampler: Sampler
    con: sql.Connection

    def __init__(self, sampler:Sampler, con:sql.Connection, commit_every:int = 1_000):
        self.sampler = sampler
        self.con = con
        self.idx = self.sampler.get_random()
        self.commit_every = commit_every

    def _run_step(self, i):
        rates, total_rate = self.sampler.get_rates(self.idx)
        r = (1.0 - np.random.uniform(low=0, high=1)) * total_rate
        rate_picked = r
        done = False
        for n, rate in rates:
            r -= rate
            if r <= 0:
                second_draw = (1.0 - np.random.uniform(low=0, high=1))
                self.time += np.log(1.0/second_draw) * 1.0 / total_rate
                self.idx = n
                done = True
                break
        if not done:
            raise Exception(f'No rate was selected. picked: {rate_picked}, total: {total_rate}')
        self.con.execute(f'INSERT INTO kmc (step, time, idx) VALUES ({i}, {self.time}, {self.idx})')
        if i % self.commit_every == 0:
            self.con.commit()

    # noinspection SqlWithoutWhere
    def run(self, steps:int, use_tqdm:bool = True) -> None:
        self.con.execute('CREATE TABLE IF NOT EXISTS kmc (step INT PRIMARY KEY, time REAL, idx INT)')
        self.con.execute('DELETE FROM kmc')
        self.con.execute('INSERT INTO kmc (step, time, idx) VALUES (0, 0, ?)', (self.idx,))
        self.con.commit()
        if use_tqdm:
            with tqdm(total=steps, desc='Running KMC', unit='steps') as pbar:
                pbar.update(1)
                for i in range(1, steps):
                    self._run_step(i)
                    pbar.update(1)
        else:
            for i in range(1, steps):
                self._run_step(i)
        self.con.commit()

class KMCv2Runner:
    out_dir: Path
    lattice_dir: Path
    kmc_dir: Path
    cons: List[sql.Connection]
    names: List[str]
    samplers: List[DualSampler]
    n_samples: int

    ORTHOGONAL_LATTICES = {}

    @staticmethod
    def arr_str(arr: List | Tuple) -> str:
        def val_to_str(x):
            if isinstance(x, float):
                return f'{x:.2f}'
            return f'{x}'

        return '_'.join([val_to_str(x) for x in arr])

    def __init__(self, out_dir: Path, dims: List[List[int]], lattice_vectors: List[List[float]],
                 init_params: List[Tuple[float, float]], saddle_params: List[Tuple[float, float]], temp: float | Iterable[float],
                 lattice_type: str | OrthogonalLattice.LatticeType = 'SC', prefactor: float = 1e13, n_samples=100):
        count = len(dims)
        try:
            _ = (e for e in temp)
        except TypeError:
            temp = [temp for _ in range(count)]
        if count != len(lattice_vectors) or count != len(init_params) or count != len(saddle_params) or count != len(temp):
            raise ValueError('All lists must have the same length')
        self.n_samples = n_samples
        self.out_dir = out_dir
        self.lattice_dir = out_dir / 'lattices'
        self.kmc_dir = out_dir / 'kmc'
        self.lattice_dir.mkdir(parents=True, exist_ok=True)
        self.kmc_dir.mkdir(parents=True, exist_ok=True)
        self.cons = []
        self.names = []
        self.samplers = []
        arr_str = KMCv2Runner.arr_str
        for i in range(count):
            name = f'{temp[i]}K{arr_str(dims[i])}D{arr_str(lattice_vectors[i])}V{arr_str(init_params[i])}I{arr_str(saddle_params[i])}S'
            self.names.append(name)
            con = sql.connect(self.lattice_dir / f'{name}.db')
            self.cons.append(con)
            t = (lattice_type, tuple(dims[i]), tuple(lattice_vectors[i]))
            if t not in KMCv2Runner.ORTHOGONAL_LATTICES:
                KMCv2Runner.ORTHOGONAL_LATTICES[t] = OrthogonalLattice(lattice_type, dims[i], lattice_vectors[i])
            lattice = KMCv2Runner.ORTHOGONAL_LATTICES[t]
            lattice.write_to_db(con)
            sampler = DualSampler(lattice, init_params[i], saddle_params[i], temp[i], prefactor)
            sampler.write_to_db(con)
            self.samplers.append(sampler)
            con.execute('CREATE TABLE IF NOT EXISTS msd_linear (run INT PRIMARY KEY,slope REAL, intercept REAL, r REAL, max_dt REAL)')

    @staticmethod
    def _run_kmc(args) -> (int, int, float, float, float):
        random.seed(uuid.uuid4().bytes)
        sampler, lattice_db_name, kmc_db_name, steps, n_samples, lat, run = args
        with sql.connect(lattice_db_name) as lattice_con, sql.connect(kmc_db_name) as kmc_con:
            kmc = KMCv2(sampler, kmc_con, commit_every=100_000)
            kmc.run(steps,use_tqdm=False)
            analyzer = KMCv2Analyzer(kmc_con, lattice_con)
            analyzer.calculate_positions(use_tqdm=False)
            analyzer.calculate_msds(500_000)
            times, msds = analyzer.get_msds()
            index = np.searchsorted(times, times[-1]/n_samples)
            linreg = stats.linregress(times[:index+1], msds[:index+1])
            return lat, run, linreg.slope, linreg.intercept, linreg.rvalue, times[index]


    def run(self, steps:int, runs: Iterable, use_tqdm:bool = True) -> None:
        args = []
        for i in range(len(self.samplers)):
            for run in runs:
                args.append((self.samplers[i], self.lattice_dir / f'{self.names[i]}.db',
                             self.kmc_dir / f'{self.names[i]}-{run}.db', steps, self.n_samples, i, run))
        shuffle(args)
        with mp.Pool() as pool:
            pbar = None
            if use_tqdm:
                pbar = tqdm(total=len(args), desc='Running KMCs', unit='runs')
            for res in pool.imap_unordered(KMCv2Runner._run_kmc, args):
                if pbar is not None:
                    pbar.update(1)
                lat, run, slope, intercept, r, max_dt = res
                con = self.cons[lat]
                con.execute('INSERT INTO msd_linear (run, slope, intercept, r, max_dt) VALUES (?, ?, ?, ?, ?)', (run, slope, intercept, r, max_dt))
            for con in self.cons:
                con.commit()
            if pbar is not None:
                pbar.close()

class KMCv2Viewer:
    def __init__(self, data_dir: Path, dims: List[List[int]], lattice_vectors: List[List[float]],
                 init_params: List[Tuple[float, float]], saddle_params: List[Tuple[float, float]],
                 temp: float | Iterable[float]):
        count = len(dims)
        try:
            _ = (e for e in temp)
        except TypeError:
            temp = [temp for _ in range(count)]
        if count != len(lattice_vectors) or count != len(init_params) or count != len(saddle_params) or count != len(temp):
            raise ValueError('All lists must have the same length')
        self.lattice_dir = data_dir / 'lattices'
        arr_str = KMCv2Runner.arr_str
        self.cons = []
        self.names = []
        for i in range(count):
            name = f'{temp[i]}K{arr_str(dims[i])}D{arr_str(lattice_vectors[i])}V{arr_str(init_params[i])}I{arr_str(saddle_params[i])}S'
            self.names.append(name)
            con = sql.connect(self.lattice_dir / f'{name}.db')
            self.cons.append(con)


    def get_slopes(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=['dims','init_sd','saddle_sd','slope', 'temp', 'R_val'])
        for con in self.cons:
            cur = con.cursor()
            cur.execute('SELECT slope, r FROM msd_linear ORDER BY run')
            slopes_arr = []
            r_vals_arr = []
            for row in cur.fetchall():
                slopes_arr.append(row[0])
                r_vals_arr.append(row[1])
            sampler_params = cur.execute('SELECT * FROM dual_sampler_params').fetchall()
            lattice_params = cur.execute('SELECT * FROM lattice_params').fetchall()
            dims = None
            for key, val in lattice_params:
                if key == 'dimensions':
                    dims = tuple([int(x) for x in str(val).split()])
                    break

            if dims is None:
                continue

            init_sd = None
            saddle_sd = None
            temp = None
            for key, val in sampler_params:
                if key == 'saddle_sd':
                    saddle_sd = float(val)
                    continue
                if key == 'init_sd':
                    init_sd = float(val)
                    continue
                if key == 'temp':
                    temp = float(val)
            if saddle_sd is None or init_sd is None or temp is None:
                print(f'Invalid sd for {dims}')
                continue
            for slope, r in zip(slopes_arr, r_vals_arr):
                data = {
                    'dims' : dims,
                    'init_sd' : init_sd,
                    'saddle_sd' : saddle_sd,
                    'slope': slope,
                    'temp' : temp,
                    'R_val' : r
                }
                df.loc[len(df)] = data
        return df
