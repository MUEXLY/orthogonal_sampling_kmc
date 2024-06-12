from typing import Iterable, List
from typing import Tuple
from kmc import OrthogonalLattice, SaddleEnergies, InitialEnergies, KineticMonteCarlo
from analysis import DumpAnalyzer
from os import mkdir, path
import numpy as np
import csv
import sys


def get_file_list(dump_pattern: str, dump_iterable: Iterable[int]) -> List[str]:
    """
    Get a list of dump files from a pattern and an iterable.
    :param dump_pattern: Pattern to match.
    :param dump_iterable: Iterable to match.
    :return: List of dump files.
    """
    dump_files = []
    for i in dump_iterable:
        dump_files.append(dump_pattern.replace('*', str(i)))
    return dump_files


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


def create_dirs_in_pattern(pattern: str, iterable: Iterable[int]) -> None:
    """
    Create directories in a pattern.
    :param iterable: Iterable in pattern.
    :param pattern: Pattern to create directories in.
    """
    if pattern.find('/') == -1:
        return
    # Cut off filename
    pattern = pattern[:pattern.rfind('/')]

    possible_dirs = []
    for i in iterable:
        possible_dirs.append(pattern.replace('*', str(i)))

    for possible_dir in possible_dirs:
        if not path.exists(possible_dir):
            mkdir(possible_dir)


def main(args: dict) -> None:
    """
    Run the analysis.
    """
    # Unpack arguments
    verbose = args['verbose']
    csv_file = args['csv_file']
    file_encoding = args['file_encoding']
    init_mean = args['init_mean']
    init_sd = args['init_sd']
    saddle_mean = args['saddle_mean']
    saddle_sd = args['saddle_sd']
    lattice_type = args['lattice_type']
    lattice_dimensions = args['lattice_dimensions']
    lattice_spacing = args['lattice_spacing']
    temperature = args['temperature']
    dump_pattern = args['dump_pattern']
    dump_iterable = args['dump_count']
    dump_every_n_steps = args['dump_every']
    num_steps = args['num_steps']
    num_points_analysis = args['num_points_analysis']
    analysis_file_out = args['analysis_file_out']
    v_print = print if verbose else lambda *a, **k: None
    if init_mean is not None and init_sd is not None and saddle_mean is not None and saddle_sd is not None:
        v_print("Using provided initial and saddle parameters.")
        init_params = (init_mean, init_sd)
        saddle_params = (saddle_mean, saddle_sd)
    else:
        v_print("Getting parameters from CSV file...")
        init_params, saddle_params = get_params_from_csv(csv_file, file_encoding)
    if init_mean is not None:
        init_params = (init_mean, init_params[1])
    if init_sd is not None:
        init_params = (init_params[0], init_sd)
    if saddle_mean is not None:
        saddle_params = (saddle_mean, saddle_params[1])
    if saddle_sd is not None:
        saddle_params = (saddle_params[0], saddle_sd)
    v_print("Done!")
    v_print(f"Initial parameters: {init_params}")
    v_print(f"Saddle parameters: {saddle_params}")
    v_print("Initializing lattice...")
    lattice = OrthogonalLattice(lattice_type, lattice_dimensions,
                                np.zeros(len(lattice_dimensions)) + 1 * lattice_spacing)
    lattice.get_adjacency_matrix(strict=True)
    v_print("Initializing energies...")
    init_energies = InitialEnergies(lattice.num_sites, np.random.normal, init_params)
    saddle_energies = SaddleEnergies(lattice.get_adjacency_matrix(), np.random.normal, saddle_params,
                                     init_energies=init_energies)
    v_print("Initializing KMC...")
    kmc = KineticMonteCarlo(lattice, init_energies, saddle_energies, temperature)
    file_list = get_file_list(dump_pattern, dump_iterable)
    create_dirs_in_pattern(dump_pattern, dump_iterable)
    v_print("Running KMC...")
    kmc.run_many(num_steps, file_list, file_encoding, dump_every_n_steps, verbose)
    # kmc.run(steps=num_steps,vacancy_dump_file=open(file_list[0], 'w', encoding=file_encoding), dump_lat_every=dump_every_n_steps, verbose=verbose)
    v_print("Done!")
    v_print('Starting Analysis...')
    analyzer = DumpAnalyzer(file_list, num_points_analysis, file_encoding)
    v_print('Done!')
    v_print('Writing analysis to file...')
    with open(analysis_file_out, 'w', encoding=file_encoding) as file:
        analyzer.to_file(file)
    v_print('Analysis written to file.')
    v_print('Done!')


def parse_argument_file(file_path: str) -> dict:
    pass


def get_default_args() -> dict:
    return {
        'temperature': 1000,
        'csv_file': './csv/1.csv',
        'dump_pattern': f'./dumps_compressed/dump.*.xz',
        'count': 200,
        'dump_every': 100,
        'num_steps': 100_000,
        'file_encoding': 'utf-8',
        'lattice_type': 'fcc',
        'lattice_dimensions': (10, 10, 10),
        'lattice_spacing': 1.0,
        'num_points_analysis': 20000,
        'analysis_file_out': f'./analysis_files/analysis.txt',
        'verbose': True,
        'init_mean': None,
        'init_sd': None,
        'saddle_mean': None,
        'saddle_sd': None
    }


def print_help():
    print('Usage: python run.py [options]')
    print('Options:')
    print('-t <temperature>: Temperature of the system.')
    print('-csv <csv_file>: CSV file containing the initial and saddle energies.')
    print('-dump_pattern <dump_pattern>: Pattern for the dump files.')
    print('-count <count>: Number of dump files to create.')
    print('-dump_every <dump_every>: Dump every n steps.')
    print('-num_steps <num_steps>: Number of steps to run the simulation.')
    print('-file_encoding <file_encoding>: Encoding of the files.')
    print('-lattice_type <lattice_type>: Type of lattice.')
    print('-lattice_dimensions <lattice_dimensions>: Dimensions of the lattice.')
    print('-lattice_spacing <lattice_spacing>: Spacing of the lattice.')
    print('-num_points_analysis <num_points_analysis>: Number of points to analyze.')
    print('-analysis_file_out <analysis_file_out>: File to write the analysis to.')
    print('-verbose: Print verbose output.')
    print('-init-mean <init_mean>: Initial mean energy.')
    print('-init-sd <init_sd>: Initial standard deviation.')
    print('-saddle-mean <saddle_mean>: Saddle mean energy.')
    print('-saddle-sd <saddle_sd>: Saddle standard deviation.')
    print('\n')
    print('Defaults:')
    args = get_default_args()
    for key, value in args.items():
        print(f'{key}: {value}')
    exit(0)


def parse_command_line_arguments() -> dict:
    arguments = sys.argv[1:]
    arguments_dict = get_default_args()
    for i, arg in enumerate(arguments):
        if arg == '-h':
            print_help()
        elif arg == '-verbose':
            arguments_dict['verbose'] = True

        if arg == '-t':
            arguments_dict['temperature'] = float(arguments[i + 1])
        elif arg == '-csv':
            arguments_dict['csv_file'] = arguments[i + 1]
        elif arg == '-dump_pattern':
            arguments_dict['dump_pattern'] = arguments[i + 1]
        elif arg == '-count':
            arguments_dict['count'] = int(arguments[i + 1])
        elif arg == '-dump_every':
            arguments_dict['dump_every'] = int(arguments[i + 1])
        elif arg == '-num_steps':
            arguments_dict['num_steps'] = int(arguments[i + 1])
        elif arg == '-file_encoding':
            arguments_dict['file_encoding'] = arguments[i + 1]
        elif arg == '-lattice_type':
            arguments_dict['lattice_type'] = arguments[i + 1]
        elif arg == '-lattice_dimensions':
            arguments_dict['lattice_dimensions'] = tuple(int(x) for x in arguments[i + 1].split(','))
        elif arg == '-lattice_spacing':
            arguments_dict['lattice_spacing'] = float(arguments[i + 1])
        elif arg == '-num_points_analysis':
            arguments_dict['num_points_analysis'] = int(arguments[i + 1])
        elif arg == '-analysis_file_out':
            arguments_dict['analysis_file_out'] = arguments[i + 1]
        elif arg == '-init-mean':
            arguments_dict['init_mean'] = float(arguments[i + 1])
        elif arg == '-init-sd':
            arguments_dict['init_sd'] = float(arguments[i + 1])
        elif arg == '-saddle-mean':
            arguments_dict['saddle_mean'] = float(arguments[i + 1])
        elif arg == '-saddle-sd':
            arguments_dict['saddle_sd'] = float(arguments[i + 1])
    arguments_dict['dump_count'] = range(arguments_dict['count'])
    # delete the count argument
    del arguments_dict['count']
    return arguments_dict


if __name__ == '__main__':
    args = parse_command_line_arguments()
    print('Arguments:')
    for key, value in args.items():
        print(f'{key}: {value}')
    main(args)
    exit(0)
