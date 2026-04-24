from pathlib import Path
from utils import strip_split
import numpy as np
import pandas as pd
import logging

potpaw_PBE_path = Path('/storage/group/xvw5285/default/ATAT/vasp_pots/potpaw_PBE.54/')

logger = logging.getLogger('VaspUtils')

class VaspFile:
    def __init__(self, file_path: Path = None, contents_str: str = None):
        self.name = self.__class__.__name__
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
        logger.debug(f'{self.name}: loaded lines from string')
    
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
        logger.debug(f'{self.name}: loaded lines from {file_path}')
    
    def write_to_file(self, dest_path: Path, overwrite_path=False):
        """Writes lines with trailing newline characters to a given filepath."""
        with open(dest_path, 'w') as d:
            for l in self.lines:
                d.write(l+'\n')
            logger.debug(f'{self.name}: wrote lines to {dest_path}')
        if overwrite_path:
            self.path = dest_path
            logger.debug(f'{self.name}: updated current path to {dest_path}')

    def append_line(self, new_line: str):
        """Append a line and overwrite the existing file if it exists."""
        self.lines.append(new_line)
        if self.path:
            self.write_to_file(self.path)

    def overwrite_line(self, line_number: int, new_line: str):
        """Overwrite a line at a given line number and overwrite the existing file if it exists."""
        self.lines[line_number] = new_line
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
    def __init__(self, file_path=None, contents_str=None):
        super().__init__(file_path=file_path, contents_str=contents_str)
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
        if self.lattice_type in ['fcc_prim']:
            self.lattice_parameters['a'] = (4*self.volume)**(1/3)
        elif self.lattice_type in ['bcc_conv', 'fcc_conv']:
            self.lattice_parameters['a'] = self.volume**(1/3)
        elif self.lattice_type in ['fcc_super', 'bcc_super']:
            ax, bx, cx = self.supercell_shape
            if [ax]*3 == [ax, bx, cx]:
                self.lattice_parameters['a'] = (self.scale_factor*self.lattice_vectors)[0][0] / ax
            else:
                raise ValueError(f'[{ax}x{bx}x{cx}] Supercell shape unsupported for fcc, bcc supercells')
        else:
            raise ValueError(f'[{self.lattice_type}] Lattice type is unsupported')

        logger.debug(f'{self.name}: updated supercell properties')
    
    def get_species(self):
        species = strip_split(self.lines[5])
        amounts = strip_split(self.lines[6], item_type=int)
        species_list = []
        for i, s in enumerate(species):
            species_list += [s]*amounts[i]
        return species, amounts, species_list
    
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
        super().load_from_string(contents_str)
        # load POTCAR for each directory name
        potcar_lines: list[str] = []
        for potcar_dir in self.lines:
            potcar_path = potpaw_PBE_path / potcar_dir / 'POTCAR'
            try:
                with open(potcar_path) as src_p:
                    potcar_lines += src_p.readlines()
            except:
                raise FileNotFoundError(f'[{potcar_path}] File does not exist')
        # remove trailing \n characters
        potcar_lines = [l.strip('\n') for l in potcar_lines]
        logger.debug(f'{self.name}: loaded lines for {self.lines}')
        self.lines = potcar_lines

class VaspOutcar(VaspFile):
    def get_energy(self):
        """Scrub through OUTCAR until the last energy is read."""
        last_line_w_energy = None
        for line in self.lines:
            if 'energy(sigma->0)' in line:
                last_line_w_energy = line
        energy = strip_split(last_line_w_energy)[-1]
        return float(energy)
    
    def get_magmom(self, poscar: VaspPoscar):
        """Scrub through OUTCAR until the magnetization is found and return average moment by ion species."""
        last_line_w_magnetization = None
        for i, line in enumerate(self.lines):
            if 'magnetization (x)' in line:
                last_line_w_magnetization = i
        # decomposed magnetic moments found
        if last_line_w_magnetization:
            species, amounts, species_list = poscar.get_species()
            magmoms = {s: [] for s in species}
            for i, s in enumerate(species_list):
                # take 'tot' value
                magmom = strip_split(self.lines[last_line_w_magnetization+3+i], item_type=float)[-1]
                magmoms[s].append(magmom)
            magmoms = {s: sum(magmoms[s]) / len(magmoms[s]) for s in species}
        else:
            magmoms = None
        return magmoms
    
class VaspContcar(VaspPoscar):
    def __init__(self, file_path=None, contents_str=None):
        super().__init__(file_path=file_path, contents_str=contents_str)
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
class VaspIncar(VaspText):            
    def remove_line(self, vasp_kw: str):
        current_line = self.check_by_keyword(vasp_kw)
        if current_line is not None:
            super().remove_line(current_line[0])

class VaspPoscar(VaspText):
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
"""