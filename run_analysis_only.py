from typing import List, Iterable
from analysis import DumpAnalyzer


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


def main():
    dump_pattern = './dumps_compressed/dump.*.xz'
    dump_iterable = range(64)
    num_points_analysis = 200000
    analysis_file_out = 'analysis.txt'
    dump_files = get_file_list(dump_pattern, dump_iterable)
    print('Analyzing dumps...')
    dump_analyzer = DumpAnalyzer(dump_files, num_points_analysis, read_every_n_steps=10)
    print('Writing analysis to file...')
    with open(analysis_file_out, 'w') as file:
        dump_analyzer.to_file(file)


if __name__ == '__main__':
    main()
