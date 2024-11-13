import lzma

from dataclasses import dataclass
from typing import List, IO

@dataclass
class AdditionalAnalyzer:
    additional_files:List[IO]
    out_file:IO

    def __init__(self, files:List[IO], out:IO):
        assert len(files) > 0
        self.additional_files = files
        self.out_file = out

    def analyze(self):
        for file in self.additional_files:
            file.seek(0)
        line = ''
        while line != 'Barriers Crossed\n':
            line = self.additional_files[0].readline()
            for file in self.additional_files[1:]:
                file.readline()
            self.out_file.write(line)

        barriers = {}
        for file in self.additional_files:
            while True:
                line = file.readline()
                if line == '':
                    break

                line = line.split()
                bar, count = float(line[0]), int(line[1])
                if bar not in barriers:
                    barriers[bar] = 0
                barriers[bar] += count

        for key, val in barriers.items():
            self.out_file.write(f'{str(key)}\t{str(val)}\n')