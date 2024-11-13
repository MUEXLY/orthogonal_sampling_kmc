from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, IO
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
import lzma


@dataclass
class DumpAnalyzer:
    """
    Class for analyzing LAMMPS dump files.
    """

    dump_files: Union[List[str], None]
    num_points: int
    times: np.ndarray
    data: np.ndarray
    nth_step: int
    standard_deviation: Union[np.ndarray, None] = None
    line_of_best_fit: Tuple[float, float] = None

    def __init__(self, dump_files: Union[Tuple[str, ...], List[str], None] = None, num_points: int = 1000,
                 file_encoding: str = 'utf-8', import_data_file: Union[IO, None] = None, single_thread: bool = False,
                 read_every_n_steps: int = 1):
        """
        Initialize the DumpAnalyzer.
        :param dump_files: List of LAMMPS dump files to process.
        :param num_points: Number of points for interpolation (default is 1000).
        :param file_encoding: Encoding of the dump files (default is 'utf-8').
        :param import_data_file: File to import data from (default is None).
        :param single_thread: Whether to use a single thread (default is False).
        :param read_every_n_steps: Read every n steps (default is 1).
        """
        self.nth_step = read_every_n_steps
        if import_data_file is not None:
            self._init_from_file(import_data_file)
            return
        if dump_files is None:
            raise ValueError("No dump files provided")
        self.dump_files = list(dump_files)  # Convert to a list if provided as a tuple
        self.num_points = num_points
        if single_thread:
            self.times, self.data, self.standard_deviation = self.process_files_one_thread(file_encoding=file_encoding)
        else:
            self.times, self.data, self.standard_deviation = self.process_files(file_encoding=file_encoding)

    def _init_from_file(self, file: IO):
        """
        Initialize the DumpAnalyzer from a file.
        :param file: File to import data from.
        """
        first_line = file.readline()
        file_version = -1
        if first_line == "Time\tAverage Squared Displacement\tStandard Deviation\n":
            file_version = 2
        elif first_line == "Time\tAverage Squared Displacement\n":
            file_version = 1
        else:
            raise ValueError("Invalid file format")
        if file_version == 1:
            time = []
            data = []
            for line in file:
                time.append(float(line.split()[0]))
                data.append(float(line.split()[1]))
            self.times = np.array(time)
            self.data = np.array(data)
            self.dump_files = None
            self.num_points = len(self.times)
            self.standard_deviation = None
            return
        if file_version == 2:
            time = []
            data = []
            std_dev = []
            for line in file:
                time.append(float(line.split()[0]))
                data.append(float(line.split()[1]))
                std_dev.append(float(line.split()[2]))
            self.times = np.array(time)
            self.data = np.array(data)
            self.dump_files = None
            self.num_points = len(self.times)
            self.standard_deviation = np.array(std_dev)
            return

    def to_file(self, file: IO):
        """
        Write the data to a file.
        :param file: File to write to.
        """
        file.write(f"Time\tAverage Squared Displacement\tStandard Deviation\n")
        for time, data, std_dev in zip(self.times, self.data, self.standard_deviation):
            file.write(f"{time}\t{data}\t{std_dev}\n")

    @staticmethod
    def _process_single_file(file: IO, nth_step: int) -> Tuple[np.ndarray, np.ndarray]:
        displacements = []
        dimensions = 0
        times = []
        bounds = None
        first_position = None

        for line in file:
            if "ITEM: BOX BOUNDS" in line:
                # count the amount of times pp shows up, that's our dimensions
                dimensions = line.count("pp")
                bound_lines = [next(file) for _ in range(dimensions)]
                bounds = [float(x) for line in bound_lines for x in line.split() if x != '0.0']
                bounds = np.array(bounds)
                assert len(bounds) == dimensions
                break
        if bounds is None:
            raise ValueError("No bounds found in dump file")
        file.seek(0)
        current_time_step = -1
        for line in file:
            if "ITEM: TIMESTEP" in line:
                current_time_step = current_time_step + 1
                if current_time_step % nth_step != 0:
                    continue
                time_step, time = next(file).split()
                time_step = int(time_step)
                time = float(time)
                # Skip until we are at the position
                for _ in range(3 + dimensions):
                    next(file)
                labels = next(file).split()
                if len(labels) != 4 + dimensions*2:
                    # 4 for "ITEM:" "ATOMS" "id" "type" and 2*dimensions for position and period
                    raise ValueError("Invalid position line")
                position_line = next(file)
                position_line = position_line.split()
                # Discard the first two values (atom id and atom type)
                position_line = position_line[2:]
                # convert to floats
                position_line = list(map(float, position_line))
                # First half is the atom position, second half is the periodic boundary conditions
                position = np.array(position_line[:dimensions])
                offset = np.array(position_line[dimensions:])
                position = position + offset * bounds

                if first_position is None:
                    first_position = position

                displacement = position - first_position
                displacements.append(displacement)
                times.append(time)

        displacements = np.array(displacements)
        displacements = np.sum(displacements ** 2, axis=1)
        return displacements, np.array(times)

    @staticmethod
    def read_dump_file(file_path: str, file_encoding: str = 'utf-8', nth_step: int = 1) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Read a LAMMPS dump file and return squared displacements and times.
        :param file_path: Path to the LAMMPS dump file.
        :param file_encoding: Encoding of the dump file (default is 'utf-8').
        :param nth_step: Read every nth step (default is 1).
        :return: Tuple containing squared displacements and times.
        """
        if not file_path.endswith(".xz"):
            with open(file_path, 'r', encoding=file_encoding) as file:
                return DumpAnalyzer._process_single_file(file, nth_step)
        with lzma.open(file_path, 'rt', encoding=file_encoding) as file:
            return DumpAnalyzer._process_single_file(file, nth_step)

    def _fit_best_line(self, step_size: int) -> float:
        """
        We will fit the best line that uses at least 50% of the data.
        :param step_size: step size for fitting the line
        :return: r value of the line
        """
        offset = 0
        best_r = -float('inf')
        best_line = None
        while True:
            r = np.corrcoef(self.times[offset:], self.data[offset:])[0, 1]
            if r > best_r:
                A = np.vstack([self.times[offset:], np.ones(len(self.times[offset:]))]).T
                best_line = np.linalg.lstsq(A, self.data[offset:], rcond=None)[0]
                best_r = r
            offset += step_size
            if offset >= len(self.times) / 2:
                break
        m, c = best_line
        self.line_of_best_fit = (m, c)
        return best_r

    def fit_line(self, r_threshold: Union[float, None] = None, step_size: int = 1) -> float:
        """
        Fit a line to the linear portion of the average squared displacement.
        :param r_threshold: if set will fit the first line with r > r_threshold
        :param step_size: step size for fitting the line
        :return: r value of the line
        """
        if r_threshold is None:
            return self._fit_best_line(step_size)
        offset = 0

        while True:
            r = np.corrcoef(self.times[offset:], self.data[offset:])[0, 1]
            if r > r_threshold:
                A = np.vstack([self.times[offset:], np.ones(len(self.times[offset:]))]).T
                m, c = np.linalg.lstsq(A, self.data[offset:], rcond=None)[0]
                self.line_of_best_fit = (m, c)
                return float(r)
            offset += step_size
            if offset >= len(self.times):
                raise ValueError("No line found with r > r_threshold")

    def generate_plot(self, title: str = 'Average Squared Displacement vs. Time',
                      output_filename: Union[str, None] = None):
        """
        Generate a plot of the average squared displacement vs. time.
        :param output_filename: Name of the output plot file (default is None).
        :param title: Title of the plot
        """
        if self.standard_deviation is None:
            plt.plot(self.times, self.data, label='Average Squared Displacement')
        else:
            plt.errorbar(self.times, self.data, yerr=self.standard_deviation, label='Average Squared Displacement',
                         fmt='-o')
        if self.line_of_best_fit is not None:
            plt.plot(self.times, self.line_of_best_fit[0] * self.times + self.line_of_best_fit[1],
                     label="Line of Best Fit")
        plt.xlabel("Time")
        plt.ylabel("Average Squared Displacement")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if output_filename is not None:
            plt.savefig(output_filename)
            plt.clf()
            return
        plt.show()
        plt.clf()

    @staticmethod
    def interpolate_data(data: np.ndarray, times: np.ndarray, num_points: int, interval: float) -> np.ndarray:
        """
        Interpolate data to obtain a regular time interval.
        :param data: List of data to interpolate.
        :param times: List of corresponding times.
        :param num_points: Number of points for interpolation.
        :param interval: Time interval for interpolation.
        :return: List of interpolated data.
        """
        interpolated_data = np.interp(np.linspace(0, interval * num_points, num_points), times, data)
        return interpolated_data

    @staticmethod
    def _get_last_time_from_file(file: IO) -> float:
        """
        Get the last time from a dump file.
        :param file: File to read from.
        :return: Last time.
        """
        time = -float('inf')
        for line in file:
            if "ITEM: TIMESTEP" in line:
                time = float(next(file).split()[1])
        return time

    @staticmethod
    def _open_file_for_last_line(filename: str, encoding) -> float:
        if filename.endswith(".xz"):
            with lzma.open(filename, 'rt', encoding=encoding) as file:
                return DumpAnalyzer._get_last_time_from_file(file)
        with open(filename, 'r', encoding=encoding) as file:
            return DumpAnalyzer._get_last_time_from_file(file)

    def get_min_interval(self, encoding: str = 'utf-8') -> float:
        """
        Calculate an appropriate time interval based on the shortest dump file.
        :return: Calculated time interval.
        """
        min_end_time = float("inf")
        args = [(file_path, encoding) for file_path in self.dump_files]
        with Pool() as pool:
            end_times = pool.starmap(DumpAnalyzer._open_file_for_last_line, args)
        for end_time in end_times:
            min_end_time = min(min_end_time, end_time)
        return min_end_time / self.num_points

    @staticmethod
    def _process_file_single_thread(file_path: str, num_points: int, interval: float, file_encoding: str,
                                    nth_step: int):
        squared_displacements, times = DumpAnalyzer.read_dump_file(file_path, file_encoding, nth_step)
        interpolated_displacements = DumpAnalyzer.interpolate_data(squared_displacements, times, num_points, interval)
        return interpolated_displacements

    def process_files(self, interval: float = None, file_encoding: str = 'utf-8') -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the dump files, calculate the average squared displacement, and return times and data.
        :param interval: Time interval for interpolation (default is None).
        :param file_encoding: Encoding of the dump files (default is 'utf-8').
        :return: Tuple containing times, average squared displacement and the standard deviation.
        """
        if interval is None:
            interval = self.get_min_interval(encoding=file_encoding)
        print(f"Using interval of {interval}")
        all_squared_displacements = []
        times = None
        with Pool() as pool:
            args = [(file_path, self.num_points, interval, file_encoding, self.nth_step) for file_path in
                    self.dump_files]
            all_squared_displacements = pool.starmap(DumpAnalyzer._process_file_single_thread, args)
        # Calculate the average squared displacement
        print("Calculating average squared displacement...")
        average_squared_displacement = np.mean(all_squared_displacements, axis=0)
        standard_deviation = np.std(all_squared_displacements, axis=0)
        times = np.linspace(0, interval * self.num_points, self.num_points)
        return times, average_squared_displacement, standard_deviation

    def process_files_one_thread(self, interval: float = None, file_encoding: str = 'utf-8') -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the dump files, calculate the average squared displacement, and return times and data.
        :param interval: Time interval for interpolation (default is None).
        :param file_encoding: Encoding of the dump files (default is 'utf-8').
        :return: Tuple containing times, average squared displacement and the standard deviation.
        """
        if interval is None:
            interval = self.get_min_interval(encoding=file_encoding)
        all_squared_displacements = []
        times = None
        args = [(file_path, self.num_points, interval, file_encoding, self.nth_step) for file_path in self.dump_files]
        for arg in args:
            all_squared_displacements.append(DumpAnalyzer._process_file_single_thread(*arg))
        # Calculate the average squared displacement
        average_squared_displacement = np.mean(all_squared_displacements, axis=0)
        standard_deviation = np.std(all_squared_displacements, axis=0)
        times = np.linspace(0, interval * self.num_points, self.num_points)
        return times, average_squared_displacement, standard_deviation
