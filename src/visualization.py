from pathlib import Path
import argparse
from vasp_file import VaspPoscar, VaspContcar
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
from math import isclose

def reflect_periodicity(coords: list[float]):
    """Given coordinates for an ion, find all ions at the next boundary to complete the cell."""
    new_rows, branches = set(), []
    coords = tuple(coords) # ensures immutability and is hashable (can be added to sets)

    # there are zeros -> add extra ions at opposite boundaries to reflect PBCs
    if any(isclose(c, 0, abs_tol=1e-2) for c in coords):
        branches.append(coords)
        while len(branches):
            # load most previously saved branch
            current_coords = branches[-1]
            branches.pop(-1)
            num_zeros_in_coords = len([c for c in current_coords if isclose(c, 0, abs_tol=1e-2)])
            
            # only one coord to flip in current coords
            if num_zeros_in_coords == 1:
                # flip the 0 and save it
                new_coords = list(current_coords)
                for c in range(3):
                    if isclose(new_coords[c], 0, abs_tol=1e-2):
                        new_coords[c] += 1
                new_coords = tuple(new_coords)
                new_rows.add(new_coords)

            # more than one zero -> flip each 0 individually, add to branches
            else:
                for c in range(3):
                    if isclose(current_coords[c], 0, abs_tol=1e-2):
                        new_coords = list(current_coords)
                        new_coords[c] += 1
                        new_coords = tuple(new_coords)
                        branches.append(new_coords)
                        new_rows.add(new_coords)
        
    return new_rows

def build_dataframe(contcar: VaspContcar, ref_poscar: VaspPoscar = None):
    """Loads the positions, species type, and defect type into a Pandas DataFrame using CONTCAR_0."""
    df = contcar.load_ion_positions(as_df=True)

    # add species column to df
    _, _, species_list = contcar.load_species()
    df['species'] = species_list

    # add defect column to df
    df['defect'] = [False]*len(df)
    if ref_poscar is not None:
        # determine defects
        ref_ion_positions = [el.tolist() for el in ref_poscar.load_ion_positions()]
        contcar_ion_positions = [el.tolist() for el in contcar.load_ion_positions()]
        defect_type = len(ref_ion_positions) - len(contcar_ion_positions)

        # vacancy (add species)
        if defect_type == 1:
            for ion in ref_ion_positions:
                if ion not in contcar_ion_positions:
                    df.loc[len(df)] = ion + ['vac', True]

        # interstitial (change defect to True)
        elif defect_type == -1:
            for ion in contcar_ion_positions:
                if ion not in ref_ion_positions:
                    x, y, z = ion
                    idx = df.index[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)][0]
                    df.loc[idx, 'defect'] = True

    return df

def update_dataframe(prev_df: pd.DataFrame, next_contcar: VaspContcar):
    # update ion positions
    new_ion_positions = next_contcar.load_ion_positions()
    new_df = prev_df.copy()
    for i, ion in enumerate(new_ion_positions):
        new_df.loc[new_df.index[i], 'x'] = ion[0]
        new_df.loc[new_df.index[i], 'y'] = ion[1]
        new_df.loc[new_df.index[i], 'z'] = ion[2]
    return new_df

if __name__ == '__main__':

    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Path to directory containing relaxed cell with point defect. Must contain steps/')
    parser.add_argument('--defective', type=str, default=False, help='(Default: False) Flag indicating the presence of a point defect in the CONTCAR files')
    args = parser.parse_args()

    # define/check paths
    main_dir = Path(args.dir).resolve()
    steps_dir = main_dir / 'steps'
    assert steps_dir.exists(), f'[{steps_dir}] Directory does not exist'
    if args.defective:
        defect_free_poscar_path = Path(main_dir).parent / 'POSCAR'
        assert defect_free_poscar_path.exists(), f'[{defect_free_poscar_path}] Defect-free POSCAR file does not exist'
        ref_poscar = VaspPoscar(defect_free_poscar_path)
    else:
        ref_poscar = None

    # load ion positions in files with CONTCAR_X format
    steps = {}
    for fp in steps_dir.iterdir():
        fn = fp.name
        if fn[:8] == 'CONTCAR_':
            steps.update({int(fn[8:]): fp})
    if len(steps) == 0:
        raise FileNotFoundError(f'[{steps_dir}] Could not find any files following CONTCAR_X format')
    
    # initialize cell and update it over time
    steps[0] = build_dataframe(VaspContcar(steps[0]), ref_poscar=ref_poscar)
    for i in range(1, len(steps)):
        steps[i] = update_dataframe(steps[i-1], VaspContcar(steps[i]))
    
    # add ions at periodic boundary conditions
    for i in range(len(steps)):
        # add atoms at boundaries
        for idx, row in steps[i].iterrows():
            *coords, species, defect = list(row)
            new_coords = reflect_periodicity(coords)
            # append extra ions to DataFrame
            for ion in list(new_coords):
                steps[i].loc[len(steps[i])] = list(ion) + [species, defect]

    # add colors
    for i in range(len(steps)):
        colors = []
        for idx, row in steps[i].iterrows():
            *coords, species, defect = list(row)
            if defect:
                colors.append('red')
            else:
                colors.append('blue')
        steps[i]['color'] = colors

    # individual frames in animation (CONTCAR files)
    frames = [
        go.Frame(
            name=str(i),
            data=go.Scatter3d(
                x=steps[i]['x'], 
                y=steps[i]['y'], 
                z=steps[i]['z'], 
                mode='markers',
                marker = dict(
                    color = steps[0]['color'],
                )
            ),
            layout = go.Layout(title_text = f'Step: {i+1}'),
        )
        for i in range(len(steps))
    ]

    # layout with title and buttons
    layout = go.Layout(
        title_text = 'Step: 1',
        updatemenus = [dict(
            type = 'buttons',
            buttons = [
                dict(
                    label = 'Play',
                    method = 'animate',
                    args = [
                        None,
                        dict(
                            frame = dict(duration = 50, redraw = False),
                            transition = dict(duration = 0),
                            fromcurrent = True,
                        ),
                    ]
                ),
                
            ]
        )],
        scene_camera=dict(
            eye = dict(x=0, y=2, z=0)
        )
    )

    # create figure
    fig = go.Figure(
        data = frames[0].data,
        layout = layout,
        frames = frames,
    )   

    # save as html for viewing in browser
    pio.write_html(fig, file=steps_dir / 'relax.html')