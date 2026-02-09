from pathlib import Path
import argparse
from vasp_file import VaspContcar
import plotly.express as px
import pandas as pd

if __name__ == '__main__':

    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Path to directory with VASP CONTCAR steps')
    args = parser.parse_args()

    # load ion positions in files with CONTCAR_X format
    steps_dir = Path(args.dir).resolve()
    steps = {}
    for fp in steps_dir.iterdir():
        fn = fp.name
        if fn[:8] == 'CONTCAR_':
            contcar = VaspContcar(fp)
            contcar.load_ion_positions(as_df=True)
            steps.update({int(fn[-1]): contcar.ion_positions})
    if len(steps) == 0:
        raise FileNotFoundError(f'[{steps_dir}] Could not find any files with CONTCAR_X format')
    
    # plot CONTCAR
    fig = px.scatter_3d(steps[0])
    fig.show()