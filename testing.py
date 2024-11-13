import numpy as np
from os.path import exists
from os import mkdir
import ovito
import ovito.io
import ovito.modifiers
from kmc import OrthogonalLattice, KineticMonteCarlo, InitialEnergies, SaddleEnergies


def get_dominant_structure(dump_file_name):
    modifier = ovito.modifiers.PolyhedralTemplateMatchingModifier()
    modifier.structures[ovito.modifiers.PolyhedralTemplateMatchingModifier.Type.SC].enabled = True

    pipeline = ovito.io.import_file(dump_file_name)
    pipeline.modifiers.append(modifier)

    data = pipeline.compute(0)
    num_sites = len(data.particles['Particle Identifier'][...])
    for attribute, count in dict(data.attributes).items():
        if count != num_sites or 'PolyhedralTemplateMatching' not in attribute:
            continue
        return attribute.split('.')[-1]


def main():
    dimensions = (8, 8, 8)
    lattice_vector = (1, 1, 1)
    structure_types = ['SC', 'BCC', 'FCC']

    if not exists('./dumps'):
        mkdir('./dumps')

    for structure_type in structure_types:

        print('----------------------------------------------------------------------')
        print(f'Checking {structure_type}...')

        seed = 0
        np.random.seed(seed)
        # first, create bulk crystals and then check that the structure is correct

        print(f'\n- Creating bulk {structure_type} structure...')
        lat = OrthogonalLattice(structure_type, dimensions, lattice_vector)

        # initialize simulation with bulk crystal

        initial_parameters = (0.0, 0.1)
        saddle_parameters = (1.0, 0.1)

        init_energy = InitialEnergies(num_sites=lat.num_sites, generator_function=np.random.normal,
                                      function_args=initial_parameters)
        saddle_energy = SaddleEnergies(adjacency_matrix=lat.get_adjacency_matrix(), generator_function=np.random.normal,
                                       function_args=saddle_parameters, init_energies=init_energy)

        kmc = KineticMonteCarlo(lat, init_energy, saddle_energy, temp=800, rand_seed=seed)
        kmc.testing = True

        # calculate structures from simulation

        dump_file_name = f'./dumps/{structure_type.lower()}.dump'
        with open(dump_file_name, 'w') as file:
            kmc.dump(file)
        print('\t• Calculating local structures with polyhedral template matching...')
        detected_structure = get_dominant_structure(dump_file_name)
        if structure_type != detected_structure:
            raise ValueError(f'{structure_type} structure created, but {detected_structure} was detected')
        print(f'\t• {detected_structure} successfully detected')

        # check adjacency matrix for simulation for coodination numbers

        print('\n- Checking coordination numbers from adjacency matrix...')
        try:
            adj = kmc.adjacency_matrix
            print(f'\t• {structure_type} has the correct number of neighbors ({np.sum(adj, axis=0)[0]:.0f})')
        except ValueError:
            print(f'• {structure_type} does not have the correct number of neighbors')
            kmc.testing = False
            adj = kmc.adjacency_matrix
            print(f'• coordination numbers = {np.sum(adj, axis=0)}')
            raise

        # next, check that saddle point matrix symmetric and properly distributed
        # and that initial matrix is properly distributed as well

        print('\n- Checking saddle point and initial energy arrays...')

        symmetric = np.all((kmc.saddle_energies == kmc.saddle_energies.T).flatten())

        if not symmetric:
            raise ValueError(f'saddle energies matrix not symmetric')

        mean_initial, std_initial = np.mean(kmc.initial_energies), np.std(kmc.initial_energies)

        if not np.isclose(mean_initial, initial_parameters[0], atol=0.01, rtol=0.1):
            raise ValueError(f'actual mean initial {mean_initial:.3f} is not close to desired {initial_parameters[0]}')

        print(f'\t• mean initial = {mean_initial:.3f}, desired = {initial_parameters[0]}')

        if not np.isclose(std_initial, initial_parameters[1], atol=0.01, rtol=0.1):
            raise ValueError(f'actual std initial {std_initial:.3f} is not close to desired {initial_parameters[1]}')

        print(f'\t• std initial = {std_initial:.3f}, desired = {initial_parameters[1]}')

        flattened = kmc.saddle_energies.flatten()
        nonzero_flattened = flattened[flattened != 0]

        mean_saddle, std_saddle = np.mean(nonzero_flattened), np.std(nonzero_flattened)

        if not np.isclose(mean_saddle, saddle_parameters[0], atol=0.01, rtol=0.1):
            raise ValueError(f'actual saddle initial {mean_saddle:.3f} is not close to desired {saddle_parameters[0]}')

        print(f'\t• saddle initial = {mean_saddle:.3f}, desired = {saddle_parameters[0]}')

        if not np.isclose(std_saddle, saddle_parameters[1], atol=0.01, rtol=0.1):
            raise ValueError(f'actual std saddle {std_saddle:.3f} is not close to desired {saddle_parameters[1]}')

        print(f'\t• std saddle = {std_saddle:.3f}, desired = {saddle_parameters[1]}')

        kmc_file_name = f'dumps/{structure_type.lower()}_run.dump'
        print(f'\n- Performing some KMC steps, saving to {kmc_file_name}...')
        with open(kmc_file_name, 'w') as file:
            kmc.run(steps=1000, vacancy_dump_file=file)

    print('----------------------------------------------------------------------')
    print('\nAll checks successful! \N{grinning face}\n')


if __name__ == '__main__':
    main()
