from pathlib import Path
from utils import strip_split
import numpy as np
import pandas as pd

potpaw_PBE_path = Path('/storage/group/xvw5285/default/ATAT/vasp_pots/potpaw_PBE.54/')

class VaspFile:
    def __init__(self, file_path: Path = None, contents_str: str = None):
        self.lines: list[str] = None
        self.path: Path = None
        if contents_str:
            self.load_from_string(contents_str)
        if file_path:
            self.load_from_file(file_path)

    def load_from_string(self, contents_str: str):
        """Loads lines from a string representation."""
        lines = contents_str.split('\n')
        # remove elements like '\n'
        lines = [l.strip() for l in lines if l != '']
        # remove elements like '  '
        lines = [l for l in lines if l.strip() != '']
        self.lines = lines
    
    def load_from_file(self, file_path: Path):
        """Loads lines from a file and updates path as this filepath."""
        if not file_path.exists():
            raise FileNotFoundError
        self.path = file_path
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # remove any \n characters
        lines = [l.strip('\n') for l in lines]
        # remove elements like '  '
        lines = [l for l in lines if l.strip() != '']
        self.lines = lines
    
    def write_to_file(self, dest_path: Path):
        """Writes lines with trailing newline characters to a given filepath."""
        with open(dest_path, 'w') as d:
            for l in self.lines:
                d.write(l+'\n')

    def append_line(self, new_line: str):
        """Append a line and overwrite the existing file if it exists."""
        self.lines.append(new_line)
        if self.path:
            self.write_to_file(self.path)

    def overwrite_line(self, line_number: int, new_line: str):
        """Overwrite a line at a given line number (starting from 1) and overwrite the existing file if it exists."""
        self.lines[line_number+1] = new_line
        if self.path:
            self.write_to_file(self.path)

class VaspIncar(VaspFile):
    def check_by_tag(self, tag: str):
        """Return the line index and line if a given INCAR tag already exists in lines."""
        for i, l in enumerate(self.lines):
            current_tag = strip_split(l, sep='=')[0]
            if tag in current_tag:
                return i, l
        return None

    def append_line(self, new_line: str):
        """Append a line, or overwrite an existing one if the INCAR tag already exists, and overwrite the existing file if it exists."""
        new_line_tag = strip_split(new_line, sep='=')[0]
        current_line = self.check_by_tag(new_line_tag)
        if current_line is None:
            super().append_line(new_line)
        else:
            self.overwrite_line(current_line[0], new_line)

class VaspPoscar(VaspFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_supercell_properties()

    def update_supercell_properties(self):
        # obtain supercell symmetry/shape information from comment line of POSCAR
        self.lattice_type, self.supercell_shape = None, None
        comment_line = strip_split(self.lines[0])
        for val in comment_line:
            if val in ['fcc_prim', 'bcc_conv', 'hcp_prim', 'fcc_conv', 'fcc_super', 'bcc_super', 'hcp_super']:
                self.lattice_type = val
            elif val in ['1x1x1', '2x2x2', '3x3x3', '4x4x4', '3x3x2']:
                self.supercell_shape = val.split('x')
                self.supercell_shape = [float(v) for v in self.supercell_shape] 
            else:
                raise ValueError(f'[{val}] Unrecognized keyword in POSCAR comment line')
        if self.lattice_type is None:
            raise ValueError(f'[{self.path}] POSCAR is missing lattice type keyword in comment line')

        # obtain scale factor
        self.scale_factor = float(self.lines[1].strip())

        # obtain lattice vectors
        self.lattice_vectors = []
        for l in self.lines[2:5]:
            lv = strip_split(l, item_type=float)
            self.lattice_vectors.append(lv)
        self.lattice_vectors = np.array(self.lattice_vectors)

        # calculate volume
        self.volume = np.linalg.det(self.scale_factor * self.lattice_vectors)

        # calculate lattice parameters
        self.lattice_parameters = {}
    
    def load_from_string(self, contents_str):
        super().load_from_string(contents_str)
        self.update_supercell_properties()

    def load_from_file(self, file_path):
        super().load_from_file(file_path)
        self.update_supercell_properties()

    def append_line(self, new_line):
        super().append_line(new_line)
        self.update_supercell_properties()

    def overwrite_line(self, line_number, new_line):
        super().overwrite_line(line_number, new_line)
        self.update_supercell_properties()

class VaspKPoints(VaspFile):
    pass
         
class VaspPotcar(VaspFile):
    def load_from_string(self, contents_str):
        """Iterates over list of directory names (e.g., ['Co', 'Ni_pv']) and loads corresponding POTCAR files with trailing newline characters removed."""
        # load directory names
        lines = super().load_from_string(contents_str)
        # load POTCAR for each directory name
        potcar_lines = []
        for potcar_dir in lines:
            potcar_path = potpaw_PBE_path / potcar_dir / 'POTCAR'
            try:
                with open(potcar_path) as src_p:
                    potcar_lines += src_p.readlines()
            except:
                raise FileNotFoundError(f'[{potcar_path}] File does not exist')
        # remove trailing \n characters
        potcar_lines = [l.strip('\n') for l in potcar_lines]
        return potcar_lines

class VaspOutcar(VaspFile):
    def get_energy(self):
        last_line_w_energy = None
        for line in self.lines:
            if 'energy(sigma->0)' in line:
                last_line_w_energy = line
        energy = strip_split(last_line_w_energy)[-1]
        return float(energy)
    
class VaspContcar(VaspPoscar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mtime = 0

    def check_updated(self):
        """Updates lines from a file if the modification time has been changed."""
        new_mtime = self.path.stat().st_mtime
        if new_mtime != self.mtime:
            self.mtime = new_mtime
            self.load_from_file(self.path)
            return True
        else:
            return False

"""
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from utils import strip_split

potpaw_PBE_path = Path('/storage/group/xvw5285/default/ATAT/vasp_pots/potpaw_PBE.54/')
semicore = ['Ni', 'Fe', 'Cr']

class VaspFile:
    "A input/output VASP file."
    def __init__(self, path: Path):
        self.path = path
        self.exists = False
        self.update_existence()

    def update_existence(self):
        self.exists = self.path.exists()

vasp_file_registry: dict[str, VaspFile] = {}
def register_study(cls):
    vasp_file_registry[cls.__name__] = cls
    return cls


@register_study
class VaspBinary(VaspFile):
    "A binary VASP file which can not be read with open()."
    def write_to_file(self, dest_path: Path):
        assert self.exists, f'[{self.path}] File does not exist'
        shutil.copy(self.path, dest_path)

@register_study
class VaspText(VaspFile):
    "A VASP file."
    def __init__(self, path: Path):
        super().__init__(path)
        self.lines = None
        self.update_existence()

    def update_existence(self):
        super().update_existence()
        if self.exists:
            self.load_from_file()

    def load_from_file(self):
        with open(self.path, 'r') as f:
            self.lines = f.readlines()
        self.exists = True

    def load_from_string(self, contents_str: str):
        lines = contents_str.split('\n')
        # remove single \n
        lines = [l.strip() for l in lines if l != '']
    
    def write_to_file(self, dest_path: Path):
        with open(dest_path, 'w') as d:
            d.writelines(self.lines)
        self.exists = True

    def overwrite_line(self, line_idx: int, new_line: str):
        self.lines[line_idx] = new_line
        self.write_to_file(self.path)

    def remove_line(self, line_idx: int):
        self.lines.pop(line_idx)
        self.write_to_file(self.path)

    def add_line(self, line_idx: int, new_line: str):
        self.lines.insert(line_idx, new_line)
        self.write_to_file(self.path)

    def append_line(self, new_line: str):
        self.lines.append(new_line)
        self.write_to_file(self.path)

    def remove_file_suffix(self):
        if len(self.path.suffix) == 0:
            return
        else:
            new_file_path = self.path.parent / self.path.name
            assert not new_file_path.exists(), f'[{self.path}] Can not rename file, it already exists'
            self.path.unlink()
            self.path = new_file_path
            self.write_to_file(new_file_path)

@register_study
class VaspIncar(VaspText):
    def check_by_keyword(self, vasp_kw: str):
        for i, l in enumerate(self.lines):
            current_kw = strip_split(l, sep='=')[0]
            if vasp_kw in current_kw:
                return i, l
        return None
            
    def add_line(self, line_idx: int, new_line: str):
        # overwrite keyword if it exists rather than inserting
        new_line_kw = strip_split(new_line, sep='=')[0]
        current_line = self.check_by_keyword(new_line_kw)
        if current_line is None:
            super().add_line(line_idx, new_line)
        else:
            self.overwrite_line(current_line[0], new_line)

    def append_line(self, new_line: str):
        # overwrite keyword if it exists rather than appending
        new_line_kw = strip_split(new_line, sep='=')[0]
        current_line = self.check_by_keyword(new_line_kw)
        if current_line is None:
            super().append_line(new_line)
        else:
            self.overwrite_line(current_line[0], new_line)

    def remove_line(self, vasp_kw: str):
        current_line = self.check_by_keyword(vasp_kw)
        if current_line is not None:
            super().remove_line(current_line[0])


@register_study
class VaspPoscar(VaspText):
    def decode_comment(self):
        lattice_type, supercell_shape = None, None
        comment_line = strip_split(self.lines[0])
        for val in comment_line:
            if val in ['fcc_prim', 'bcc_conv', 'hcp_prim', 'fcc_conv', 'fcc_super', 'bcc_super', 'hcp_super']:
                lattice_type = val
            elif val in ['1x1x1', '2x2x2', '3x3x3', '4x4x4', '3x3x2']:
                supercell_shape = val.split('x')
                supercell_shape = [float(v) for v in supercell_shape] 
            else:
                raise ValueError(f'[{val}] Unrecognized keyword in POSCAR comment line')
        if lattice_type is None:
            raise ValueError(f'[{self.path}] POSCAR is missing lattice type keyword in comment line')
        return lattice_type, supercell_shape

    def load_scale_factor(self):
        return float(self.lines[1].strip())

    def load_lattice_vectors(self):
        lattice_vectors = []
        for l in self.lines[2:5]:
            lv = strip_split(l, item_type=float)
            lattice_vectors.append(lv)
        return np.array(lattice_vectors)
    
    def load_species(self):
        species = strip_split(self.lines[5])
        amounts = strip_split(self.lines[6], item_type=int)
        species_list = []
        for s in species:
            for a in amounts:
                species_list += [s]*a
        return species, amounts, species_list

    def load_ion_positions(self, as_df=False):
        _, amounts, _ = self.load_species()
        if as_df:
            ion_positions = pd.DataFrame(columns=['x', 'y', 'z'])
            for l in self.lines[8:8+sum(amounts)]:
                ion_positions.loc[len(ion_positions)] = np.array(strip_split(l, item_type=float))
        else:
            ion_positions = []
            for l in self.lines[8:8+sum(amounts)]:
                ion_positions.append(np.array(strip_split(l, item_type=float)))
        return ion_positions
            
    def check_by_position(self, position: list[float]):
        ion_positions = self.load_ion_positions()
        for i, l in enumerate(ion_positions):
            check_arr = np.array(position)
            if all(np.isclose(l, check_arr, atol=1e-3)):
                return i+8, l

@register_study            
class VaspPotcar(VaspText):
    def __init__(self, path: Path, poscar: VaspPoscar):
        self.species, _, _ = poscar.load_species()
        with open(path, 'w') as p:
            for s in self.species:
                if s in semicore:
                    s += '_pv'
                potcar_path = potpaw_PBE_path / s / 'POTCAR'
                try:
                    with open(potcar_path) as src_p:
                        p.writelines(src_p.readlines())
                except:
                    raise FileNotFoundError(f'[{potcar_path}] File does not exist')
                p.write('\n\n')
        super().__init__(path)

@register_study
class VaspOutcar(VaspText):
    def get_energy(self):
        last_line_w_free_energy = None
        for line in self.lines:
            if 'free energy' in line:
                last_line_w_free_energy = line
        energy = strip_split(last_line_w_free_energy)[4]
        return float(energy)

    def get_magmom(self):
        last_line_w_magnetization = None
        for i, line in enumerate(self.lines):
            if 'magnetization' in line:
                last_line_w_free_magnetization = i
        magmom = strip_split(self.lines[last_line_w_free_magnetization+4])[-1]
        return float(magmom)

@register_study
class VaspContcar(VaspPoscar):
    def __init__(self, path: Path):
        super().__init__(path)
        self.mtime = 0

    def check_updated(self):
        # copy CONTCAR if it exists and it has been updated
        if self.exists:
            new_mtime = self.path.stat().st_mtime
            if new_mtime != self.mtime:
                self.mtime = new_mtime
                self.load_from_file()
                return True
            else:
                return False
        else:
            self.update_existence()

vasp_input_file_types = {
        'INCAR': VaspIncar,
        'POSCAR': VaspPoscar,
        'KPOINTS': VaspText,
        'POTCAR': VaspPotcar,
}
vasp_output_file_types = {
        'CHG':      VaspText, 
        'CHGCAR':   VaspText, 
        'CONTCAR':  VaspContcar, 
        'DOSCAR':   VaspText, 
        'EIGENVAL': VaspText, 
        'IBZKPT':   VaspText, 
        'OSZICAR':  VaspText, 
        'OUTCAR':   VaspOutcar, 
        'PCDAT':    VaspText, 
        'PROCAR':   VaspText, 
        'REPORT':   VaspText, 
        'WAVECAR':  VaspBinary, 
        'XDATCAR':  VaspText,
}
vasp_file_types = vasp_input_file_types | vasp_output_file_types
"""