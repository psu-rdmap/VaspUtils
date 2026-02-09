from pathlib import Path
from cell import Cell
from vasp_file import vasp_file_types, vasp_output_file_types

def strip_split(s: str, sep=None, item_type=None):
    s = s.strip()
    if item_type is int:
        return [int(x) for x in s]
    elif item_type is float:
        return [float(x) for x in s]
    elif item_type is None:
        return s.split(sep)
    else:
        raise ValueError(f'[{item_type}] Invalid item type. Choose None, int, or float')

def tilps(list_str: list[str], sep: str = ' '):
    s = ''
    for l in list_str:
        s += str(l) + sep
    return s