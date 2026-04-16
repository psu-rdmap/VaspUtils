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

import argparse, yaml
from pathlib import Path
from studies import study_registry

def main():
    # load input file
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file')
    args = parser.parse_args()

    input_fp = Path(args.input).resolve()
    assert input_fp.exists(), f'[{input_fp}] File does not exist'

    with open(input_fp, 'r') as f:
        input_params: dict = yaml.safe_load(f)

    # start a study
    study_params = input_params['study']['parameters']
    study = study_registry[study_params['type']](input_params)

main()