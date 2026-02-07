from pathlib import Path
import numpy as np
import shutil, time

potpaw_PBE_path = Path('/storage/group/xvw5285/default/ATAT/vasp_pots/potpaw_PBE.54/')
semicore = ['Ni', 'Fe', 'Cr']

def strip_split(s: str, sep=None):
    s = s.strip()
    return s.split(sep)

def strip_split_float(s: str):
    s = strip_split(s)
    return [float(x) for x in s]

def strip_split_int(s: str):
    s = strip_split(s)
    return [int(x) for x in s]

def tilps(list_str: list[str], sep: str = ' '):
    s = ''
    for l in list_str:
        s += str(l) + sep
    return s

class VaspFile:
    """A input/output VASP file."""
    def __init__(self, path: Path):
        self.path = path
        self.exists = False
        self.update_existence()

    def update_existence(self):
        self.exists = self.path.exists()

class VaspBinary(VaspFile):
    """A binary VASP file which can not be read with open()."""
    def write_to_file(self, dest_path: Path):
        assert self.exists, f'[{self.path}] File does not exist'
        shutil.copy(self.path, dest_path)

class VaspText(VaspFile):
    """A VASP file."""
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

class VaspIncar(VaspText):
    def check_by_keyword(self, vasp_kw: str):
        for i, l in enumerate(self.lines):
            if vasp_kw in l:
                return i, l
        return None
            
    def add_line(self, line_idx: int, new_line: str):
        # overwrite keyword if it exists rather than inserting
        new_line_kw = strip_split(new_line, sep='=')[0]
        current_line = self.check_by_keyword(new_line_kw)
        if current_line is None:
            super().add_line(line_idx, new_line)
        else:
            self.overwrite_line(current_line[0], current_line[1])

    def append_line(self, new_line: str):
        # overwrite keyword if it exists rather than appending
        new_line_kw = strip_split(new_line, sep='=')[0]
        current_line = self.check_by_keyword(new_line_kw)
        if current_line is None:
            super().append_line(new_line)
        else:
            self.overwrite_line(current_line[0], current_line[1])

class VaspPoscar(VaspText):
    def decode_comment(self):
        lattice_type, supercell_shape = None, None
        comment_line = strip_split(self.lines[0])
        for val in comment_line:
            if val in ['fcc_prim', 'bcc_conv', 'hcp_prim', 'fcc_conv', 'fcc_super', 'bcc_super', 'hcp_super']:
                lattice_type = val
            elif val in ['2x2x2', '3x3x3', '3x3x2']:
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
            lv = strip_split_float(l)
            lattice_vectors.append(lv)
        return np.array(lattice_vectors)
    
    def load_species(self):
        species = strip_split(self.lines[5])
        amounts = strip_split_int(self.lines[6])
        species_list = []
        for s in species:
            for a in amounts:
                species_list += [s]*a
        return species, amounts, species_list

    def check_by_position(self, position: list[float]):
        _, amounts, _ = self.load_species()
        for i, l in enumerate(self.lines[8:8+sum(amounts)]):
            arr_1 = np.array(strip_split_float(l))
            arr_2 = np.array(position)
            if all(np.isclose(arr_1, arr_2, atol=1e-3)):
                return i+8, l
                
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
