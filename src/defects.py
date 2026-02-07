from pathlib import Path
import numpy as np
import argparse, os
from ase.units import kJ
from cell import Cell, copy_from_cell, cleanup_vasp_output
from vasp_file import tilps, VaspText, VaspIncar, VaspPoscar

# user input
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='Path to directory with relaxed INCAR, POSCAR, KPOINTS')
parser.add_argument('--incar', type=str, default=None, help='(Default: INCAR in --dir) Path to specific INCAR file to load')
parser.add_argument('--cores', type=int, default=8, help='(Default: 8) Number of cores to run VASP with')
parser.add_argument('--defect', type=str, help='Type of point defect to add to the cell (vac, int)')
args = parser.parse_args()

num_cpus_available = int(os.environ.get("SLURM_NTASKS", 1))
assert args.cores > 0, f'Number of requested cores must be greater than 0'
assert args.cores <= num_cpus_available, f'Too many requested cores ({num_cpus_available} available)'

def point_defect(cell: Cell, defect_type: str, incar: VaspIncar = None):
    """Return the relaxed cell with a point defect added."""
    # make defect directory
    i = 0
    defect_dir = None
    while defect_dir is None:
        if (cell.dir / f'defect_{i}').exists():
            i += 1
        else:
            defect_dir = cell.dir / f'defect_{i}'
            defect_dir.mkdir()
    
    # create new cell object and copy contcar into poscar
    defect_cell = copy_from_cell(cell, defect_dir)
    if incar:
        defect_cell.set_incar(incar)
    defect_cell.contcar_to_poscar()

    # check lattice and get line index of point defect location in POSCAR
    lt, sc = defect_cell.lattice_type, defect_cell.supercell_shape
    err_str = f'[{lt}, {sc[0]}x{sc[1]}x{sc[2]}] Unsupported lattice_type and/or supercell shape for point-defect energy calculations'
    line_idx = None
    int_lines = []

    # fcc supercells
    if lt == 'fcc_super':
        if sc == [2, 2, 2]:
            line_idx = defect_cell.poscar.check_by_position([0.5, 0.5, 0.5])[0]
            int_lines.append('  0.3333333333333333  0.5000000000000000  0.5000000000000000\n')
            int_lines.append('  0.6666666666666666  0.5000000000000000  0.5000000000000000\n')
        else:
            raise ValueError(err_str)

    # bcc supercells
    elif lt == 'bcc_super':
        if sc == [2, 2, 2]:
            line_idx = defect_cell.poscar.check_by_position([0.5, 0.5, 0.5])[0]
            int_lines.append('  0.3333333333333333  0.5000000000000000  0.5000000000000000\n')
            int_lines.append('  0.6666666666666666  0.5000000000000000  0.5000000000000000\n')
        else:
            raise ValueError(err_str)

    # hcp supercells
    elif lt == 'hcp_super':
        if sc == [3, 3, 2]:
            line_idx = defect_cell.poscar.check_by_position([0.44444, 0.55555, 0.625])[0]   
            int_lines.append('  0.3333333333333333  0.5555555555555555  0.6250000000000000\n')
            int_lines.append('  0.5555555555555555  0.5555555555555555  0.6250000000000000\n')
        else:
            raise ValueError(err_str)
    else:
        raise ValueError(err_str)
       
    # insert defect into poscar
    _, species_amounts, _ = defect_cell.poscar.load_species()
    if defect_type == 'vac':
        species_amounts[0] -= 1
        defect_cell.poscar.remove_line(line_idx)
    elif defect_type == 'int':
        species_amounts[0] += 1
        defect_cell.poscar.overwrite_line(line_idx, int_lines[0])
        defect_cell.poscar.add_line(line_idx, int_lines[1])
    else:
        raise ValueError(f'[{defect_type}] Unsupported defect type. Choose from (vac, int)')
    species_amounts_line = tilps(species_amounts)
    defect_cell.poscar.overwrite_line(6, species_amounts_line+'\n')
        
    # update magmom in incar
    magmom_line = defect_cell.incar.check_by_keyword('MAGMOM')
    if magmom_line:
        m_idx, m_line = magmom_line

        # remove trailing comment if present
        m_line = m_line.split('#')[0]

        # separate at '=' and add it after MAGMOM
        m_line = m_line.split('=')
        m_line[0] = 'MAGMOM'
        m_line.insert(1, '=')

        # increase or decrease first atom count number after =
        magmoms = m_line[2].split()
        first_magmom = magmoms[0].split('*')
        first_magmom.insert(1, '*')
        first_magmom_count = int(first_magmom[0])
        if defect_type == 'vac':
            first_magmom_count -= 1
        elif defect_type == 'int':
            first_magmom_count += 1

        # update line in incar
        first_magmom[0] = str(first_magmom_count)
        first_magmom = tilps(first_magmom, sep='')
        magmoms[0] = first_magmom
        m_line.pop(2)
        m_line += magmoms
        m_line = tilps(m_line)
        defect_cell.incar.overwrite_line(m_idx, m_line+'\n')

    # final changes to incar
    lorbit_line = defect_cell.incar.check_by_keyword('LORBIT')
    if lorbit_line:
        defect_cell.incar.remove_line(lorbit_line[0])
    istart_line = defect_cell.incar.check_by_keyword('ISTART')
    if istart_line:
        defect_cell.incar.remove_line(istart_line[0])

    # run VASP
    defect_cell.run_vasp()

    return defect_cell

if __name__ == '__main__':
    relax_dir = Path(args.dir).resolve()
    assert relax_dir.exists(), f'[{relax_dir}] Directory does not exist'
    
    # create cell based on relaxed directory
    relax_cell = Cell(relax_dir, args.cores)

    # insert point defect and relax
    if args.incar:
        incar = VaspIncar(Path(args.incar).resolve())
    else:
        incar = None
    defect_cell = point_defect(relax_cell, args.defect, incar=incar)

    # print out properties of the main cell
    with open(defect_cell.dir / 'defect.out', 'w') as r:
        r.write(f'Energy: {defect_cell.energy} eV\n')
        print(f'Energy: {defect_cell.energy} eV\n')
