"""
The idea is that a user supplies one large format input file describing

1. Study details -> parameters and input files
    a. Input parameter convergence
        i. Parameter name and values
    a. Birch-Murnaghan EoS fitting
        i. Volume scaling factors
    b. Chemical potential for HEAs
        i. 
    c. Defect energetics
        i. Point defect type
        ii. Energy type

2. Calculation details -> updates to input files
    a. Preconvergence
        i. PREC, ENCUT, EDIFF, EDIFFG, KPOINTS
    b. Non-magnetic / magnetic
        i. ISPIN, MAGMOM, LORBIT
    d. High accuracy
        i. LREAL, PREC, ENCUT, EDIFF, EDIFFG, KPOINTS
    e. Atom replacement
        i. 
"""

import argparse, yaml, shutil, logging, sys
from pathlib import Path
from studies import Study, study_registry

# basic logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('debug')

def main():
    # load input file
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file')
    args = parser.parse_args()

    input_fp = Path(args.input).resolve()
    assert input_fp.exists(), f'[{input_fp}] File does not exist'

    # load user input
    with open(input_fp, 'r') as f:
        input_params: dict = yaml.safe_load(f)
    logger.debug(f'Loaded input file {input_fp}')

    # initialize a study
    study_type = input_params['study']['parameters']['type']
    study: Study = study_registry[study_type](input_params)
    logger.debug(f'Initialized study type {study_type}')
    
    # build directory tree and copy in input file
    study.build_directory()
    logger.debug(f'Built directory tree at {study.dir_path}')
    shutil.copy(input_fp, study.dir_path)

    # run vasp
    logger.debug(f'Starting VASP calculations')
    study.run_vasp()

main()