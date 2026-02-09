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

def copy_from_cell(cell: Cell, dest_dir: Path):
    """Copy the existing cells vasp files into the destination and create a new cell."""
    dest_dir.mkdir(exist_ok=True)
    for vfn in vasp_file_types.keys():
        if hasattr(cell, vfn.lower()):
            vf = getattr(cell, vfn.lower())
            if vf.exists:
                vf.write_to_file(dest_dir / vfn)
    return Cell(dest_dir, cell.cores)

def cleanup_vasp_output(cell: Cell):
    for vfn in vasp_output_file_types.keys():
        if vfn == 'OUTCAR':
            continue
        vf = getattr(cell, vfn.lower())
        cell.delete_vasp_file(vf)
    # other files
    (cell.dir / 'vaspout.h5').unlink(missing_ok=True)
    (cell.dir / 'vasprun.xml').unlink(missing_ok=True)
