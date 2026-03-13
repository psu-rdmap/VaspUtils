from copy import deepcopy
import numpy as np

def strip_split(s: str, sep=None, item_type=None):
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

def tilps(list_str: list[str], sep: str = ' '):
    s = ''
    for l in list_str:
        s += f'{l:.16f}' + sep
    return s

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