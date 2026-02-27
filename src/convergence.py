# Given an INCAR, KPOINTS, POSCAR, perform convergence studies for KPOINTS, ENCUT

import argparse, re
from pathlib import Path
from cell import Cell, copy_from_cell
from utils import strip_split
from typing import Dict
      
def convergence_study(main_cell: Cell, study_fp: Path):
    # make study directory
    i = 0
    study_dir = None
    while study_dir is None:
        if (main_cell.dir / f'convergence_{i}').exists():
            i += 1
        else:
            study_dir = main_cell.dir / f'convergence_{i}'
            study_dir.mkdir()
    
    # load study file
    with open(study_fp, 'r') as sf:
        study_type = sf.readline().strip()
        assert study_type in ['KPOINTS', 'ENCUT'], f'[{study_type}] Study type not recognized. Must be KPOINTS or ENCUT'
        study_vals = strip_split(sf.readline(), sep=' ')
    
    # construct cells
    study_cells: Dict[str, Cell] = dict()
    for val in study_vals:
        val_dir = study_dir / val
        study_cells.update({val: copy_from_cell(main_cell, val_dir)})

    # overwrite INCAR or KPOINTS
    for val, cell in study_cells.items():
        if study_type == 'KPOINTS':
            cell.kpoints.overwrite_line(3, re.sub('x', ' ', val))
        elif study_type == 'ENCUT':
            cell.incar.append_line(f'ENCUT = {val}\n')
    
    # relax each cells
    energies = []
    for val, cell in study_cells.items():
        cell.run_vasp()
        energies.append(cell.energy)
    
    # write energies
    with open(study_dir / 'energies.out', 'w') as out:
        lines = [str(e)+'\n' for e in energies]
        out.writelines(lines)

if __name__ == '__main__':
    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Path to directory with INCAR, POSCAR, KPOINTS')
    parser.add_argument('--study', type=str, help='Path to file with input information for conducting a convergence study')
    args = parser.parse_args()

    main_dir = Path(args.dir).resolve()
    assert main_dir.exists(), f'[{main_dir}] Directory does not exist'

    study_fp = Path(args.study).resolve()
    assert study_fp.exists(), f'[{study_fp}] File does not exist'
    
    # create cell from input directory
    main_cell = Cell(main_dir)

    # run convergence study
    convergence_study(main_cell, study_fp)