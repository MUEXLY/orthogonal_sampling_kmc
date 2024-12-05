import sqlite3 as sql
import multiprocessing as mp
from typing import Tuple

import numpy as np

from tqdm import tqdm


# noinspection SqlWithoutWhere
class KMCv2Analyzer:
    con: sql.Connection

    def __init__(self, con: sql.Connection):
        self.con = con

    def calculate_positions(self, use_tqdm: bool = True):
        pbar = None
        if use_tqdm:
            total = self.con.execute('SELECT COUNT(*) FROM kmc').fetchone()[0]
            pbar = tqdm(total=total, unit='step', desc='Unrolling positions')
        bounds = [float(x) for x in self.con.execute('SELECT value FROM lattice_params WHERE key = "bounds"').fetchone()[0].split()]
        bounds = np.array(bounds)
        dims = len(bounds)
        pos_str = ', '.join([f'x{i}' for i in range(dims)])
        question_marks = ', '.join(['?' for _ in range(dims)])
        self.con.execute(f'CREATE TABLE IF NOT EXISTS positions (step INTEGER PRIMARY KEY, {pos_str})')
        self.con.execute('DELETE FROM positions')
        cur = self.con.cursor()
        cur1 = self.con.cursor()
        cur.execute('SELECT idx, step FROM kmc ORDER BY step')
        prev_pos = None
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
            if prev_pos is None:
                prev_pos = pos

            displacement = pos - prev_pos
            displacement = np.where(displacement > 0.5 * bounds, displacement - bounds, displacement)
            displacement = np.where(displacement < -0.5 * bounds, displacement + bounds, displacement)
            pos = prev_pos + displacement
            self.con.execute(f'INSERT INTO positions (step, {pos_str}) VALUES ({step}, {question_marks})', tuple(pos))
            prev_pos = pos
            if pbar is not None:
                pbar.update(1)
        self.con.commit()

    def _calculate_msd(self, delta_t: float, max_time: float, dims: int) -> Tuple[float, float, int] | None:
        pos_str = ', '.join([f'x{i}' for i in range(dims)])
        id_query = 'SELECT step from kmc WHERE time <= ? ORDER BY time DESC LIMIT 1'
        pos_query = f'SELECT {pos_str} FROM positions WHERE step = ?'
        cur = self.con.cursor()
        square_displacements = []
        current_time = 0
        current_step = None
        current_position = None
        cur.execute(id_query, (current_time,))
        row = cur.fetchone()
        if row is not None:
            current_step = row[0]
        else:
            return
        cur.execute(pos_query, (current_step,))
        row = cur.fetchone()
        if row is not None:
            current_position = np.array(row)
        else:
            return
        count = 0
        while current_time + delta_t <= max_time:
            count += 1
            current_time += delta_t
            cur.execute(id_query, (current_time,))
            row = cur.fetchone()
            if row is None:
                break
            current_step = row[0]
            cur.execute(pos_query, (current_step,))
            row = cur.fetchone()
            if row is None:
                break
            next_position = np.array(row)
            displacement = next_position - current_position
            square_displacement = np.sum(displacement**2)
            square_displacements.append(square_displacement)
            current_position = next_position
        msd = float(np.mean(square_displacements))
        sd = float(np.std(square_displacements))
        return msd, sd, count

    def _setup_msd_table(self):
        self.con.execute('CREATE TABLE IF NOT EXISTS msd (dt REAL PRIMARY KEY, msd REAL, sd REAL, count INTEGER)')
        self.con.execute('DELETE FROM msd')
        self.con.commit()

    # noinspection SqlWithoutWhere
    def calculate_msds(self, num_delta_ts: int, max_delta_t: float | None = None, use_tqdm: bool = True):
        self._setup_msd_table()
        max_time = float(self.con.execute('SELECT MAX(time) FROM kmc').fetchone()[0])
        if max_delta_t is None:
            max_delta_t = max_time
        delta_ts = np.linspace(0, max_delta_t, num_delta_ts+1)[1:]
        dims = int(self.con.execute('SELECT value FROM lattice_params WHERE key = "num_dimensions"').fetchone()[0])
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=num_delta_ts, unit='dts', desc='Calculating MSDs')
        for delta_t in delta_ts:
            ret = self._calculate_msd(delta_t, max_time, dims)
            if ret is None:
                if pbar is not None:
                    pbar.update(1)
                continue
            msd, sd, count = ret
            self.con.execute('INSERT INTO msd (dt, msd, sd, count) VALUES (?, ?, ?, ?)', (delta_t, msd, sd, count))
            if pbar is not None:
                pbar.update(1)
        self.con.commit()

    @staticmethod
    def _calculate_msds_single(args) -> Tuple[float, float, float, int] | None:
        delta_t, max_time, dims, db = args
        con = sql.connect(db)
        analyzer = KMCv2Analyzer(con)
        ret = analyzer._calculate_msd(delta_t, max_time, dims)
        if ret is None:
            return
        return delta_t, ret[0], ret[1], ret[2]


    def calculate_msds_multi(self, num_delta_ts: int, db:str, max_delta_t: float | None = None, use_tqdm: bool = True):
        self._setup_msd_table()
        max_time = float(self.con.execute('SELECT MAX(time) FROM kmc').fetchone()[0])
        if max_delta_t is None:
            max_delta_t = max_time
        delta_ts = np.linspace(0, max_delta_t, num_delta_ts + 1)[1:]
        dims = int(self.con.execute('SELECT value FROM lattice_params WHERE key = "num_dimensions"').fetchone()[0])
        args = [(delta_t, max_time, dims, db) for delta_t in delta_ts]
        self.con.commit()
        with mp.Pool() as pool:
            pbar = None
            if use_tqdm:
                pbar = tqdm(total=num_delta_ts, unit='dts', desc='Calculating MSDs')
            for ret in pool.imap_unordered(self._calculate_msds_single, args):
                if ret is None:
                    pbar.update(1)
                    continue
                dt, msd, sd, count = ret
                self.con.execute('INSERT INTO msd (dt, msd, sd, count) VALUES (?, ?, ?, ?)', (dt, msd, sd, count))
                if pbar is not None:
                    pbar.update(1)
        self.con.commit()


    def get_msds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dts = []
        msds = []
        sds = []
        counts = []
        cur = self.con.cursor()
        cur.execute('SELECT dt, msd, sd, count FROM msd ORDER BY dt')
        for row in cur:
            dts.append(row[0])
            msds.append(row[1])
            sds.append(row[2])
            counts.append(row[3])
        return np.array(dts), np.array(msds), np.array(sds), np.array(counts)