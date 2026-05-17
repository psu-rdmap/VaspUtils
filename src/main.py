import argparse, yaml, shutil, logging, sys
from pathlib import Path
from studies import Study, study_registry

# basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('VaspUtils')

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
    study_type = input_params['study']['type']
    study: Study = study_registry[study_type](input_params)
    logger.debug(f'Initialized study type {study_type}')
    
    # build directory tree and copy in input file
    study.build_directory()
    logger.debug(f'Built directory tree at {study.dir}')
    shutil.copy(input_fp, study.dir)

    # run vasp
    logger.debug(f'Starting VASP calculations')
    study.run_vasp()

main()