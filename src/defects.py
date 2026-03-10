from pathlib import Path
from cell import Cell, copy_from_cell, cleanup_vasp_output
from utils import tilps, strip_split
import numpy as np
import argparse, yaml

def adjust_poscar(cell: Cell, input_fp: Path):
    with open(input_fp, 'r') as f:
        defect_data: dict = yaml.safe_load(f)

    # get line index and position vector of target atom to be changed
    target_atom = cell.poscar.check_by_position(defect_data['position'])
    assert target_atom is not None, f'[{defect_data['position']}] Could not find atom within 0.001 of each coordinate'

    # get list detailing number of each ion species
    _, species_amounts, _ = cell.poscar.load_species()

    # vacancy -> remove target atom
    if defect_data['type'] == 'vacancy':
        species_amounts[0] -= 1
        cell.poscar.remove_line(target_atom[0])
    # interstitial -> replace target with dumbbell
    elif defect_data['type'] == 'interstitial':
        species_amounts[0] += 1
        int_coord_1 = int_coord_2 = target_atom[1].tolist()
        if defect_data['db_orientation'] == '100':
            lv_idx = 0
        elif defect_data['db_orientation'] == '010':
            lv_idx = 1
        elif defect_data['db_orientation'] == '001':
            lv_idx = 2
        else:
            raise ValueError(f'[{defect_data['db_orientation']}] Unrecognized dumbbell orientation. Expected \'100\', \'010\', or \'001\'')
        frac_spacing = defect_data['db_spacing'] / np.linalg.norm(cell.lattice_vectors[lv_idx])
        int_coord_1[lv_idx] -= frac_spacing/2
        int_coord_2[lv_idx] += frac_spacing/2
        cell.poscar.overwrite_line(target_atom[0], tilps(int_coord_1))
        cell.poscar.add_line(target_atom[0], tilps(int_coord_2))
    else:
        raise ValueError(f'[{defect_data['type']}] Unrecognized defect type. Expected \'vacancy\' or \'interstitial\'')
    
    # update number of species accordingly
    species_amounts_line = tilps(species_amounts)
    cell.poscar.overwrite_line(6, species_amounts_line+'\n')

    return defect_data

def adjust_incar(cell: Cell, defect_data: dict):       
    # update magmom (ferromagnetic ordering ONLY)
    magmom_line = cell.incar.check_by_keyword('MAGMOM')
    if magmom_line:
        # remove trailing comment (if any) get MAGMOM values
        m_line = magmom_line[1].split('#')[0]
        magmom_list = strip_split(m_line.split('=')[1])
        assert len(magmom_list) == 1, 'Only ferromagnetic initialization is supported for interstitial point defects'

        # increase/decrease first atom amount
        magmom = magmom_list[0].split('*')
        magmom.insert(1, '*')
        magmom_atom_count = int(magmom[0])
        if defect_data['defect_type'] == 'vacancy':
            magmom_atom_count -= 1
        elif defect_data['defect_type'] == 'interstitial':
            magmom_atom_count += 1
        magmom[-1] = str(magmom_atom_count)
        magmom = tilps(magmom)

        # update line in incar
        new_m_line = f'MAGMOM = {magmom}\n'
        defect_cell.incar.overwrite_line(magmom_line[0], new_m_line)

    # more changes to incar
    cell.incar.remove_line('LORBIT')
    cell.incar.remove_line('ISTART')
    cell.incar.remove_line('ICHARG')
    cell.incar.append_line('LREAL = Auto\n')
    cell.incar.append_line('ISYM = 0\n')
    cell.incar.append_line('NELM = 200\n')    

if __name__ == '__main__':
    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Path to directory of previous relaxation')
    parser.add_argument('--input', type=str, help='Path to input file with point defect details')
    parser.add_argument('--dry-run', action='store_true', help='Generate input files, but do not run VASP')
    args = parser.parse_args()

    relax_dir = Path(args.dir).resolve()
    assert relax_dir.exists(), f'[{relax_dir}] Directory does not exist'

    input_fp = Path(args.input).resolve()
    assert input_fp.exists(), f'[{input_fp}] File does not exist'
    
    # create cell based on relaxed directory
    relax_cell = Cell(relax_dir)

    # make new directory for defective cell
    i = 0
    defect_dir = None
    while defect_dir is None:
        if (relax_cell.dir / f'defect_{i}').exists():
            i += 1
        else:
            defect_dir = relax_cell.dir / f'defect_{i}'
            defect_dir.mkdir()

    defect_cell = copy_from_cell(relax_cell, defect_dir)
    defect_cell.contcar_to_poscar()
    cleanup_vasp_output(defect_cell)

    # adjust input files
    defect_data = adjust_poscar(defect_cell, input_fp)
    adjust_incar(defect_cell, defect_data)

    # run vasp
    if not args.dry_run:
        defect_cell.run_vasp()
        with open(defect_cell.dir / 'defect.out', 'w') as r:
            r.write(f'Energy: {defect_cell.energy} eV\n')
            print(f'Energy: {defect_cell.energy} eV\n')