import sqlite3 as sql
from typing import Tuple

import numpy as np
import torch as tn

from tqdm import tqdm


# noinspection SqlWithoutWhere
class KMCv2Analyzer:
    kmc_con: sql.Connection
    lattice_con: sql.Connection

    def __init__(self, kmc_con: sql.Connection, lattice_con: sql.Connection):
        self.kmc_con = kmc_con
        self.lattice_con = lattice_con

    def calculate_positions(self, use_tqdm: bool = True):
        pbar = None
        if use_tqdm:
            total = self.kmc_con.execute('SELECT COUNT(*) FROM kmc').fetchone()[0]
            pbar = tqdm(total=total, unit='step', desc='Unrolling positions')
        bounds = [float(x) for x in self.lattice_con.execute('SELECT value FROM lattice_params WHERE key = "bounds"').fetchone()[0].split()]
        bounds = np.array(bounds)
        dims = len(bounds)
        pos_str = ', '.join([f'x{i}' for i in range(dims)])
        question_marks = ', '.join(['?' for _ in range(dims)])
        self.kmc_con.execute(f'CREATE TABLE IF NOT EXISTS positions (step INTEGER PRIMARY KEY, {pos_str})')
        self.kmc_con.execute('DELETE FROM positions')
        cur = self.kmc_con.cursor()
        cur1 = self.lattice_con.cursor()
        cur.execute('SELECT idx, step FROM kmc ORDER BY step')
        prev_pos = None
        true_prev_pos = None
        first = True
        while True:
            row = cur.fetchone()
            if row is None:
                break
            idx, step = row
            cur1.execute(f'SELECT {pos_str} FROM lattice_positions WHERE id = ?', (idx,))
            row = cur1.fetchone()
            if row is None:
                break
            pos = np.array(row)
            if first:
                first = False
                true_prev_pos = pos
                prev_pos = pos

            displacement = pos - prev_pos
            displacement = np.where(displacement > 0.5 * bounds, displacement - bounds, displacement)
            displacement = np.where(displacement < -0.5 * bounds, displacement + bounds, displacement)

            true_prev_pos = true_prev_pos + displacement
            self.kmc_con.execute(f'INSERT INTO positions (step, {pos_str}) VALUES ({step}, {question_marks})', tuple(true_prev_pos))
            prev_pos = pos
            if pbar is not None:
                pbar.update(1)
        self.kmc_con.commit()

    """
    def _calculate_msd(self, delta_t: float, max_time: float, dims: int) -> Tuple[float, float, int] | None:
        pos_str = ', '.join([f'x{i}' for i in range(dims)])
        step_query = 'SELECT step from kmc WHERE time <= ? ORDER BY time DESC LIMIT 1'
        time_query = 'SELECT time from kmc WHERE step = ?'
        pos_query = f'SELECT {pos_str} FROM positions WHERE step = ?'
        cur = self.con.cursor()
        square_displacements = []
        current_step = 0
        while True:
            init_time = cur.execute(time_query, (current_step,)).fetchone()[0]
            next_time = init_time + delta_t
            if next_time > max_time:
                break
            init_pos = np.array(cur.execute(pos_query, (current_step,)).fetchone())

            next_step = cur.execute(step_query, (next_time,)).fetchone()[0]
            next_pos = np.array(cur.execute(pos_query, (next_step,)).fetchone())

            displacement = next_pos - init_pos
            square_displacement = np.sum(displacement**2)
            square_displacements.append(square_displacement)
            current_step += 1
        msd = float(np.mean(square_displacements))
        sd = float(np.std(square_displacements))
        return msd, sd, len(square_displacements)
    
    @staticmethod
    def _calculate_msds_single(args) -> Tuple[float, float, float, int] | None:
        delta_t, max_time, dims, db = args
        con = sql.connect(db)
        analyzer = KMCv2Analyzer(con)
        ret = analyzer._calculate_msd(delta_t, max_time, dims)
        if ret is None:
            return
        return delta_t, ret[0], ret[1], ret[2]
    """

    def _setup_msd_table(self):
        self.kmc_con.execute('CREATE TABLE IF NOT EXISTS msd (dt REAL PRIMARY KEY, msd REAL)')
        self.kmc_con.execute('DELETE FROM msd')
        self.kmc_con.commit()

    def _evenly_spaced_pos(self, num_points: int, max_time:float, dims:int) -> (np.ndarray, np.ndarray):
        time_points = np.linspace(0, max_time, num_points)

        cur = self.kmc_con.cursor()
        step_query = 'SELECT step, time FROM kmc WHERE time <= ? ORDER BY time'
        all_steps = cur.execute(step_query, (max_time,)).fetchall()

        steps = np.array(all_steps, dtype=[('step', int), ('time', float)])

        matched_steps = []
        for time_point in time_points:
            closest_step = steps[steps['time'] <= time_point][-1]['step']
            matched_steps.append(closest_step)

        needed_steps = set(matched_steps)
        pos_str = ', '.join([f'x{i}' for i in range(dims)])
        position_query = f'SELECT step, {pos_str} FROM positions'

        cur.execute(position_query)
        pos_map = {}
        for row in cur:
            if row[0] in needed_steps:
                pos_map[row[0]] = np.array(row[1:])

        positions = np.array([pos_map[step] for step in matched_steps])

        return time_points, positions


    def calculate_msds(self, pos_count:int = 20000):
        self._setup_msd_table()
        max_time = float(self.kmc_con.execute('SELECT MAX(time) FROM kmc').fetchone()[0])
        dims = int(self.lattice_con.execute('SELECT value FROM lattice_params WHERE key = "num_dimensions"').fetchone()[0])
        time_array, pos_array = self._evenly_spaced_pos(pos_count, max_time, dims)
        # Code adapted from https://stackoverflow.com/questions/69738376/how-to-optimize-mean-square-displacement-for-several-particles-in-two-dimensions/69767209

        n_time = pos_array.shape[0]
        s2 = np.sum(
            np.fft.ifft(
                np.abs(np.fft.fft(pos_array, n=2*n_time, axis=0))**2, axis=0
            ).take(range(n_time), axis=0).real, axis=1
        )
        d = np.square(pos_array).sum(axis=1)

        shape_t = [n_time if ax==0 % d.ndim else 1 for ax, s in enumerate(d.shape)]
        shape_non_t = [1 if ax==0 % d.ndim else s for ax, s in enumerate(d.shape)]

        d = np.append(d, np.zeros(shape_non_t), axis = 0)

        s1 = (2*np.sum(d, axis=0).reshape(shape_non_t)
              - np.cumsum(
                    np.insert(
                        d.take(np.arange(0, n_time), axis=0),
                        0,
                        0,
                        axis=0
                    ) + np.flip(d, axis=0),
                    axis=0
                )
              ).take(
                np.arange(0, n_time),
                axis=0
              )
        msd = (s1 - 2*s2) / (n_time-np.arange(n_time).reshape(shape_t))
        dt_r = np.arange(1, n_time-1)
        msd = msd.take(dt_r, axis=0)
        time_array = time_array.take(dt_r)
        for i in range(len(dt_r)):
            self.kmc_con.execute('INSERT INTO msd (dt, msd) VALUES (?, ?)', (time_array[i], msd[i]))
        self.kmc_con.commit()



    def get_msds(self) -> Tuple[np.ndarray, np.ndarray]:
        dts = []
        msds = []
        cur = self.kmc_con.cursor()
        cur.execute('SELECT dt, msd FROM msd ORDER BY dt')
        for row in cur:
            dts.append(row[0])
            msds.append(row[1])
        return np.array(dts), np.array(msds)