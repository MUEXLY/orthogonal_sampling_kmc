# KMC

This project is a Python application that uses the Kinetic Monte Carlo method to simulate and analyze a orthogonal crystal lattice. It supports sampling energies from an arbitrary distribution but by default it is a normal distribution. It supports N dim SC lattices and 1-3 dims FCC or BCC lattices. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires some python packages.

You can install these packages using pip:

```bash
pip install -r requirements.txt
```

### Running the Application

The main entry point of the application is `run.py`. It accepts several command-line arguments to customize the simulation and analysis. Here's an example of how to run the application:

```bash
python run.py -t 1000 -csv ./csv/1.csv -dump_pattern ./dumps_compressed/dump.*.xz -count 200 -dump_every 100 -num_steps 100000 -file_encoding utf-8 -lattice_type fcc -lattice_dimensions 10,10,10 -lattice_spacing 1.0 -num_points_analysis 20000 -analysis_file_out ./analysis_files/analysis.txt -verbose
```

The output of the application will be a tab seperated file containing the analysis of the simulation.
```text
Time\tAverage Squared Displacement\tStandard Deviation\n
0.0\t0.0\t0.0\n
...
```

Here's a brief explanation of the command-line arguments:

- `-t <temperature>`: Temperature of the system.
- `-csv <csv_file>`: CSV file containing the initial and saddle energies.
- `-dump_pattern <dump_pattern>`: Pattern for the dump files.
- `-count <count>`: Number of trajectories to simulate.
- `-dump_every <dump_every>`: Dump every n steps.
- `-num_steps <num_steps>`: Number of steps to run the simulation.
- `-file_encoding <file_encoding>`: Encoding of the files.
- `-lattice_type <lattice_type>`: Type of lattice.
- `-lattice_dimensions <lattice_dimensions>`: Dimensions of the lattice.
- `-lattice_spacing <lattice_spacing>`: Spacing of the lattice.
- `-num_points_analysis <num_points_analysis>`: Number of points to analyze.
- `-analysis_file_out <analysis_file_out>`: File to write the analysis to.
- `-verbose`: Print verbose output.
- `-init-mean <init_mean>`: Initial mean energy.
- `-init-sd <init_sd>`: Initial standard deviation.
- `-saddle-mean <saddle_mean>`: Saddle mean energy.
- `-saddle-sd <saddle_sd>`: Saddle standard deviation.
- `-h`: Show help message.

This project uses the following units:
- Energy: eV
- Temperature: K
- Length: Angstrom

Due to how large dump and analysis files can get. You can add 'xz' to the end of the file name to compress the file. For example, `dump.*.xz` will compress the dump files. \
You can then use a program such as `xzcat` to view the contents of the compressed file. 

### Python modules

This project is more than just running the simulation. It contains two main modules: `kmc` and `analysis`. \
`kmc` can be used in python scripts to run simulations and provides a high degree of control. \
`analysis` can be used to analyze specifically formatted dump files produced by kmc.

There are also some notebooks in the `notebooks` directory that provide additional examples of how to use the modules.
The `sc_dims` notebook is particularly useful as it does curve fitting on multiple analysis files and makes plots. 
The `fitting_test` notebook is an example on how to use the `analysis` module to fit a curve to the data.

## Built With

- [Python](https://www.python.org/)
