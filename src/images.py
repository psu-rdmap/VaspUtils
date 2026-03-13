"""
Assumptions
- Ions should not move too far (within 0.05 fractional coordinates)
- Ions with wrapped coordinates should only have crossed the planes connected to the origin (i.e., 0.01 -> -0.01 -> 0.99, NOT 0.99 -> 1.01 -> 0.01)
"""

import argparse, shutil, yaml
from utils import tilps, wrap_coords, unwrap_coords
import numpy as np
from pathlib import Path
from vasp_file import VaspPoscar
from copy import copy

class Ion:
    def __init__(self, id0: int, r0: np.ndarray, id1: int = None, r1: np.ndarray = None, tot_dist: float = None, defect: str = None):
        # 0 -> initial POSCAR, 1 -> final POSCAR
        self.id0 = id0
        self.id1 = id1
        # position vectors
        self.r0 = r0
        self.r1 = r1
        self.tot_dist = tot_dist
        # vacancy or interstitial
        self.defect = defect
        # interpolated positions
        self.steps: list[np.ndarray] = []

if __name__ == '__main__':
    # get input file from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file with image generation input data')
    args = parser.parse_args()

    # load input file
    input_fp = Path(args.input).resolve()
    assert input_fp.exists(), f'[{input_fp}] File does not exist'

    with open(input_fp, 'r') as f:
        input_data: dict = yaml.safe_load(f)

    # check dir path
    neb_dir = Path(input_data['dir']).resolve()
    assert neb_dir.exists(), f'[{neb_dir}] Directory does not exist'

    # check poscar paths
    init_poscar_fp = Path(input_data['initial']).resolve()
    assert init_poscar_fp.exists(), f'[{init_poscar_fp}] File does not exist'

    fin_poscar_fp = Path(input_data['final']).resolve()
    assert fin_poscar_fp.exists(), f'[{fin_poscar_fp}] File does not exist'

    # load poscars and ion positions
    init_poscar = VaspPoscar(init_poscar_fp)
    fin_poscar = VaspPoscar(fin_poscar_fp)

    init_ion_pos = init_poscar.load_ion_positions()
    fin_ion_pos = fin_poscar.load_ion_positions()
    assert len(init_ion_pos) == len(fin_ion_pos), 'Expected initial and final POSCARs to have the same number of ions'

    # atoms should only cross coordinate planes connected to the origin, so unwrap them
    init_ion_pos_unwrap = unwrap_coords(init_ion_pos)
    fin_ion_pos_unwrap = unwrap_coords(fin_ion_pos)

    # create dictionary versions for matching process
    init_ion_pos_dict = {j: init_ion_pos_unwrap[j] for j in range(len(init_ion_pos_unwrap))}
    fin_ion_pos_dict = {j: fin_ion_pos_unwrap[j] for j in range(len(fin_ion_pos_unwrap))}
    
    # connect atom in final POSCAR with initial vacancy position to atom in initial POSCAR with final vacancy position
    if input_data['defect'] == 'vacancy':
        r_vac_init = np.array(input_data['defect_pos_initial'])
        r_vac_fin = np.array(input_data['defect_pos_final'])
        init_distances = {}
        fin_distances = {}
        for i in range(len(init_ion_pos)):
            r_init = init_ion_pos_unwrap[i]
            r_fin = fin_ion_pos_unwrap[i]
            init_distances.update({i: np.linalg.norm(r_vac_fin - r_init)})
            fin_distances.update({i: np.linalg.norm(r_vac_init - r_fin)})
        # search through each distance dictionary until the minimum is found
        id0 = min(init_distances, key=init_distances.get)
        id1 = min(fin_distances, key=fin_distances.get)
        r0 = init_ion_pos_unwrap[id0]
        r1 = fin_ion_pos_unwrap[id1]
        defect_ion = Ion(id0, r0, id1=id1, r1=r1, tot_dist=np.linalg.norm(r1-r0), defect='vacancy')
    # interstitial
    elif input_data['defect'] == 'interstitial':
        raise NotImplementedError
    
    # remove ions from "master" dictionaries so they do not get matched in the future
    init_ion_pos_dict.pop(id0)
    fin_ion_pos_dict.pop(id1)

    # create dictionary of ions using initial POSCAR line indices as keys
    ions = {id0: defect_ion}

    # connect all other ions in the initial and final POSCARs
    for i, r0 in init_ion_pos_dict.items():
        ion = Ion(i, r0)
        # search through unmatched positions in final poscar
        distances = {}
        for j, r1 in fin_ion_pos_dict.items():
            distances.update({j: np.linalg.norm(r1-r0)})
        closest_j = min(distances, key=distances.get)
        # remove match to ensure one-to-one correspondence
        fin_ion_pos_dict.pop(closest_j)
        # update and save current ion
        ion.id1 = closest_j
        ion.r1 = fin_ion_pos_unwrap[closest_j]
        ion.tot_dist = distances[closest_j]
        ions.update({i: ion})

    # interpolate intermediate positions for each ion
    num_steps = input_data['nimages'] + 2
    assert num_steps < 100, f"[nimages: {input_data['nimages']}] Maximum number of intermediate images is 97"
    for i, ion in ions.items():
        dr_norm = (ion.r1 - ion.r0) / ion.tot_dist
        step_dists = np.linspace(0, ion.tot_dist, num_steps).tolist()
        ion.steps = [ion.r0 + d*dr_norm for d in step_dists]

    # wrap all position vectors for all ions back to the range [0,1)
    for i, ion in ions.items():
        ion.steps = wrap_coords(ion.steps)

    # create save directory
    steps_dir = neb_dir / 'steps'
    if steps_dir.exists():
        shutil.rmtree(steps_dir)
    steps_dir.mkdir()

    # generate image POSCARs
    poscar_head = init_poscar.lines[:8]
    all_poscars: list[list[str]] = []
    for i in range(num_steps):
        current_lines = copy(poscar_head)
        for j, ion in ions.items():
            current_lines.append('  '+tilps(ions[j].steps[i].tolist(), sep='  ')+'\n')
        with open(steps_dir / f'POSCAR_{i:02d}', 'w') as p:
            p.writelines(current_lines)