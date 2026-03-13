from pathlib import Path
import argparse, yaml, natsort
from vasp_file import VaspPoscar
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
from math import isclose
from copy import deepcopy
import numpy as np
from utils import unwrap_coords

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

def build_dataframe(pos_list: list[np.ndarray]):
    pos_df = pd.DataFrame(columns=['x', 'y', 'z'])
    for pos in pos_list:
        pos_df.loc[len(pos_df)] = deepcopy(pos)
    return pos_df
    
if __name__ == '__main__':
    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file with visualization input data')
    args = parser.parse_args()

    # load input file
    input_fp = Path(args.input).resolve()
    assert input_fp.exists(), f'[{input_fp}] File does not exist'

    with open(input_fp, 'r') as f:
        input_data: dict = yaml.safe_load(f)

    # check dir path
    steps_dir = Path(input_data['dir']).resolve()
    assert steps_dir.exists(), f'[{steps_dir}] Directory does not exist'
    
    # load POSCAR/CONTCAR files using natural sorting
    steps = {fp.name: fp for fp in steps_dir.iterdir()}
    steps_keys = natsort.natsorted(steps, key=steps.get)
    steps = {i: VaspPoscar(steps[key]) for i, key in enumerate(steps_keys)}
    
    # initialize first frame to set species and color
    step0_df = steps[0].load_ion_positions()
    step0_df = unwrap_coords(step0_df)
    step0_df = build_dataframe(step0_df)
    # species
    _, _, species_list = steps[0].load_species()
    step0_df['species'] = species_list
    # colors
    color_list = []
    for spcs in species_list:
        if spcs in input_data['colors'].keys():
            color_list.append(input_data['colors'][spcs])
        else:
            color_list.append('gray')
    for i, ion in input_data['track'].items():
        r0 = np.array(ion['initial_pos'])
        distances = {}
        for j, r1 in enumerate(steps[0].load_ion_positions()):
            distances.update({j: np.linalg.norm(r1-r0)})
        closest_j = min(distances, key=distances.get)
        color_list[closest_j] = ion['color']
    step0_df['color'] = color_list
    
    # load successive frames
    frames = {0: step0_df}
    steps.pop(0)
    for i, step in steps.items():
        next_df = deepcopy(frames[0])
        next_pos_df = step.load_ion_positions()
        next_pos_df = unwrap_coords(next_pos_df)
        next_pos_df = build_dataframe(next_pos_df)
        next_df['x'] = next_pos_df['x']
        next_df['y'] = next_pos_df['y']
        next_df['z'] = next_pos_df['z']
        frames.update({i: next_df})
    
    # add ions at periodic boundary conditions
    for i in range(len(frames)):
        # add atoms at boundaries
        for idx, row in frames[i].iterrows():
            *coords, species, color = list(row)
            new_coords = reflect_periodicity(coords)
            # append extra ions to DataFrame
            for ion in list(new_coords):
                frames[i].loc[len(frames[i])] = list(ion) + [species, color]

    # individual frames in animation (CONTCAR files)
    vis_frames = [
        go.Frame(
            name=str(i),
            data=go.Scatter3d(
                x=frames[i]['x'], 
                y=frames[i]['y'], 
                z=frames[i]['z'], 
                mode='markers',
                marker = dict(
                    color = frames[0]['color'],
                )
            ),
            layout = go.Layout(title_text = f'Frame: {i+1}')
        )
        for i in range(len(frames))
    ]

    play_button = dict(
        label = 'Play',
        method = 'animate',
        args = [
            None,
            dict(
                frame = dict(duration = 10, redraw = True),
                transition = dict(duration = 10),
                fromcurrent = True,
            ),
        ]
    )

    pause_button = dict(
        label = 'Pause',
        method = 'animate',
        args = [
            [None],
            dict(
                frame = dict(duration = 0, redraw = True),
                mode = 'immediate',
                transition = dict(duration = 0),
            ),
        ]
    )

    reset_button = dict(
        label = 'Reset',
        method = 'animate',
        args = [
            [vis_frames[0].name],
            dict(
                frame = dict(duration = 0, redraw = True),
                transition = dict(duration = 0),
                mode = 'immediate',
            ),
        ]
    )

    # layout with title and buttons
    layout = go.Layout(
        title_text = 'Frame: 1',
        updatemenus = [dict(
            type = 'buttons',
            buttons = [play_button, pause_button, reset_button],
            showactive = False
        )],
    )

    # create figure
    fig = go.Figure(
        data = vis_frames[0].data,
        frames = vis_frames,
        layout = layout
    )

    # save as html for viewing in browser
    pio.write_html(fig, file=steps_dir.parent / 'relax.html', auto_play=False)