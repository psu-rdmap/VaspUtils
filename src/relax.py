from pathlib import Path
import argparse
from ase.eos import EquationOfState
from ase.units import kJ
from cell import Cell, copy_from_cell

def make_relax_dir(cell: Cell):
    i = 0
    relax_dir = None
    while relax_dir is None:
        if (cell.dir / f'relax_{i}').exists():
            i += 1
        else:
            relax_dir = cell.dir / f'relax_{i}'
            relax_dir.mkdir()
    return relax_dir

def simple_relax(cell: Cell):
    relax_dir = make_relax_dir(cell)
    current_cell = copy_from_cell(cell, relax_dir)
    current_cell.run_vasp()
    return current_cell

def eos_fit(cell: Cell):
    """Return the relaxed input cell."""
    # make relax directory
    relax_dir = make_relax_dir(cell)
    
    # scale up/down volumes
    scale_factors = [0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04]
    volumes, energies = [], []
    for i, sf in enumerate(scale_factors):
        # create new cell
        current_cell = copy_from_cell(cell, relax_dir / str(i))
        current_cell.poscar.overwrite_line(1, f'{sf*cell.scale_factor}\n')
        
        # relax and get volume and energy
        current_cell.run_vasp()
        volumes.append(current_cell.volume)
        energies.append(current_cell.energy)
    
    # fit Murnaghan equation of state
    eos = EquationOfState(volumes, energies, eos='murnaghan')
    eq_vol, eq_energy, bulk_mod = eos.fit()
    eos.plot(relax_dir / 'eos.png')

    # create cell with eq_vol and relax
    eq_cell = copy_from_cell(cell, relax_dir / 'eq')
    eq_cell.bulk_modulus = bulk_mod / kJ * 1.0e24
    eq_vol_factor = eq_vol / cell.volume
    eq_scale_factor = cell.scale_factor*(eq_vol_factor)**(1/3)
    eq_cell.poscar.overwrite_line(1, f'{eq_scale_factor}\n')
    eq_cell.run_vasp()
    
    return eq_cell
       
if __name__ == '__main__':

    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Path to directory to perform relaxation, possibly containing the INCAR, POSCAR, KPOINTS files')
    parser.add_argument('--incar', type=str, default='INCAR', help='(Default: INCAR) Specific INCAR file to load')
    parser.add_argument('--kpoints', type=str, default='KPOINTS', help='(Default: KPOINTS) Specific KPOINTS file to load')
    parser.add_argument('--poscar', type=str, default='POSCAR', help='(Default: POSCAR) Specific POSCAR file to load')
    parser.add_argument('--eos', action=argparse.BooleanOptionalAction, help='Relax many structures and fit the Murnaghan EoS')
    args = parser.parse_args()

    main_dir = Path(args.dir).resolve()
    assert main_dir.exists(), f'[{main_dir}] Directory does not exist.'
    
    # create cell based on main directory
    main_cell = Cell(main_dir, incar_fn=args.incar, poscar_fn=args.poscar, kpoints_fn=args.kpoints)

    # relax main cell
    if args.eos:
        relax_cell = eos_fit(main_cell)
        output_path = relax_cell.dir.parent / 'relax.out'
    else:
        relax_cell = simple_relax(main_cell)
        output_path = relax_cell.dir / 'relax.out'

    # print out properties of the main cell
    with open(output_path, 'w') as r:
        r.write(f'Energy: {relax_cell.energy} eV\n')
        r.write(f'Volume: {relax_cell.volume} A3\n')
        r.write(f'Lattice parameter: {relax_cell.lattice_parameter} A\n')
        r.write(f'Bulk modulus: {relax_cell.bulk_modulus} GPa\n')
        r.write(f'Magnetic moment: {relax_cell.magnetic_moment} uB\n')
    with open(output_path, 'r') as r:
        for l in r.readlines():
            print(l, end='')