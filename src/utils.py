from pathlib import Path
import shutil

def strip_split(s: str, sep=None, item_type=None):
    """Strip a string of whitespace and split it apart given a separator character."""
    s = s.strip()
    s = s.split(sep)
    if item_type is int:
        return [int(x) for x in s]
    elif item_type is float:
        return [float(x) for x in s]
    elif item_type is None:
        return s
    else:
        raise ValueError(f'[{item_type}] Invalid item type. Choose None, int, or float')

def tilps(list_vals: list, sep: str = ' ', precision: int = 16):
    """Inverse of strip(), where a list of strings are glued back together into a single string."""
    s = ''
    for l in list_vals:
        if precision == 0:
            s += str(l) + sep
        else:
            s += f'{l:.16f}' + sep
    return s.strip()

def next_path(path: Path):
    """Return a path name with the next available index appended to it (e.g., 'some_path_050')."""
    i = 0
    while i < 1000:
        new_path = path.parent / (path.name+f'_{i:003}')
        if new_path.exists():
            i += 1
        else:
            return new_path
    raise ValueError(f'[{path}] Study file path index limit reached (1000)')

def wipe_directory(path: Path):
    """Erase all files and subdirectories at a given path."""
    for p in path.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

"""
from copy import deepcopy
import numpy as np
def unwrap_coords(pos_list: list[np.ndarray]):
    new_pos_list = deepcopy(pos_list)
    for pos in new_pos_list:
        for c in range(3):
            if pos[c] > 0.95:
                pos[c] = pos[c] - 1
    return new_pos_list

def wrap_coords(pos_list: list[np.ndarray]):
    new_pos_list = deepcopy(pos_list)
    for pos in new_pos_list:
        for c in range(3):
            if pos[c] < 0:
                pos[c] = pos[c] + 1
    return new_pos_list
"""