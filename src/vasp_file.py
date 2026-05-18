from pathlib import Path
from utils import strip_split, tilps
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

POTPAW_PBE_PATH = Path('/storage/group/xvw5285/default/ATAT/vasp_pots/potpaw_PBE.54/')

logger = logging.getLogger('VaspUtils')

class VaspFile:
    alias = 'VaspFile'
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

    def insert_line(self, line_number: int, new_line: str):
        """Insert a line at a given line number and overwrite the existing file if it exists."""
        self.lines.insert(line_number, new_line)
        if self.path:
            self.write_to_file(self.path)

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
    
    def remove_line(self, line_number: int):
        """Delete a line at a given line number and overwrite the existing file if it exists."""
        self.lines.pop(line_number)
        if self.path:
            self.write_to_file(self.path)

vasp_file_registry: dict[str, VaspFile] = {}
def register_vasp_file_type(cls):
    """Registry enrollment so that Study subclasses can be instantiated by string name."""
    vasp_file_registry[cls.alias] = cls
    return cls

@register_vasp_file_type
class VaspIncar(VaspFile):
    alias = 'INCAR'
    def update_tags(self, tag_changes: list[dict]):
        for line in tag_changes:
            action = next(iter([k for k in line.keys()]))
            if action == 'Add':
                self.append_line(str(line[action]))
            elif action == 'Remove':
                self.remove_line(str(line[action]))

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

    def remove_line(self, tag: str):
        """Remove line by INCAR tag instead of line number and overwrite the existing file if it exists."""
        idx_and_line = self.check_by_tag(tag)
        if idx_and_line is not None:
            super().remove_line(idx_and_line[0])
    
    def get_magmoms(self):
        """If MAGMOM is present, return a list of magnetic moments in the same way as VaspPoscar's get_species() method."""
        magmom_line = self.check_by_tag('MAGMOM')
        if magmom_line is None:
            return None
        _, magmom_line = magmom_line
        magmoms = strip_split(strip_split(magmom_line, '=')[1])
        magmoms_list: list[str] = []
        for mag in magmoms:
            mag = mag.split('*')
            # two items corresponding to amt*val
            if len(mag) == 2:
                magmoms_list += [mag[1]]*int(mag[0])
            elif len(mag) == 1:
                magmoms_list += [mag[0]]
        return magmoms_list

    def load_magmoms_from_list(self, magmom_list: list[str]):
        unique_magmoms = dict.fromkeys(magmom_list)
        new_magmom_str = 'MAGMOM ='
        for mag in unique_magmoms.keys():
            num_mag = len([m for m in magmom_list if mag == m])
            new_magmom_str += f' {num_mag}*{mag}'
        self.append_line(new_magmom_str)

@register_vasp_file_type
class VaspPoscar(VaspFile):
    alias = 'POSCAR'
    def __init__(self, file_path=None, contents_str=None):
        super().__init__(file_path=file_path, contents_str=contents_str)
        self.update_supercell_properties()

    def load_from_string(self, contents_str):
        super().load_from_string(contents_str)
        self.update_supercell_properties()

    def load_from_file(self, file_path):
        super().load_from_file(file_path)
        self.update_supercell_properties()

    def overwrite_line(self, line_number, new_line):
        super().overwrite_line(line_number, new_line)
        self.update_supercell_properties()

    def update_supercell_properties(self):
        # obtain supercell symmetry/shape information from comment line of POSCAR
        self.lattice_type, self.supercell_shape = None, None
        comment_line = strip_split(self.lines[0])
        for val in comment_line:
            if val in ['fcc_prim', 'bcc_conv', 'fcc_conv', 'fcc_super', 'bcc_super', 'tetr_super']:
                self.lattice_type = val
            elif val in ['1x1x1', '2x2x2', '3x3x3', '4x4x4', '3x3x2', '2x2x3']:
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
            self.lattice_parameters['c'] = self.lattice_parameters['a']
        elif self.lattice_type in ['bcc_conv', 'fcc_conv']:
            self.lattice_parameters['a'] = self.volume**(1/3)
            self.lattice_parameters['c'] = self.lattice_parameters['a']
        elif self.lattice_type in ['fcc_super', 'bcc_super']:
            ax, bx, cx = self.supercell_shape
            self.lattice_parameters['a'] = (self.scale_factor*self.lattice_vectors)[0][0] / ax
            self.lattice_parameters['c'] = self.lattice_parameters['a']
        elif self.lattice_type in ['tetr_super']:
            ax, bx, cx = self.supercell_shape
            self.lattice_parameters['a'] = (self.scale_factor*self.lattice_vectors)[0][0] / ax
            self.lattice_parameters['c'] = (self.scale_factor*self.lattice_vectors)[2][2] / cx

        logger.debug(f'{self.name}: updated supercell properties')
    
    def update_scaling_factor(self, new_sf: float):
        self.overwrite_line(1, str(new_sf) + '\n')
        self.update_supercell_properties()

    def get_species(self):
        species: list[str] = strip_split(self.lines[5])
        amounts: list[int] = strip_split(self.lines[6], item_type=int)
        species_list: list[str] = []
        for i, s in enumerate(species):
            species_list += [s]*amounts[i]
        return species, amounts, species_list
    
    def get_ion_positions(self):
        _, amounts, _ = self.get_species()
        ion_positions: list[np.ndarray] = []
        for l in self.lines[8:8+sum(amounts)]:
            ion_positions.append(np.array(strip_split(l, item_type=float)))
        return ion_positions
    
    def remove_ion(self, defect_pos: np.ndarray[float], incar: VaspIncar) -> float:
        """Remove ion closest to the provided position and adjust species line and INCAR magmom line accordingly."""
        positions, distances = [], []
        for pos in self.get_ion_positions():
            positions.append(pos)
            distances.append(np.linalg.norm(pos - defect_pos))
        rm_pos_idx = distances.index(min(distances))
        self.remove_line(rm_pos_idx+8)
        # get species name corresponding to line number (e.g., 'Ni')
        species, amounts, species_list = self.get_species()
        species_name = species_list[rm_pos_idx]
        # decrement and overwrite species amount corresponding to determined name above
        amounts[species.index(species_name)] -= 1
        self.overwrite_line(6, tilps(amounts, sep='  ', precision=0))
        # remove magnetic moment from magmom list and rebuild MAGMOM line
        magmom_list = incar.get_magmoms()
        if magmom_list:
            magmom_list.pop(rm_pos_idx)
            incar.load_magmoms_from_list(magmom_list)
        # return position value of removed ion
        return positions[rm_pos_idx]

    def add_ion(self, defect_species: str, defect_pos: np.ndarray[float], incar: VaspIncar, magmom: float = None):
        """Add ion at a given position and adjust species and INCAR magmom line accordingly."""
        species, amounts, species_list = self.get_species()
        magmom_list = incar.get_magmoms()
        # species is new -> append it to end of list
        if defect_species not in species:
            species += [defect_species]
            amounts += [1]
            self.overwrite_line(5, tilps(species, sep='  ', precision=0))
            self.overwrite_line(6, tilps(amounts, sep='  ', precision=0))
            self.insert_line(len(species_list)+8, tilps(defect_pos, sep='  '))
            if magmom_list:
                magmom_list.append(f'{magmom:.2f}')
                incar.load_magmoms_from_list(magmom_list)
        # species is not new -> append it to end of specific species list
        else:
            amounts[species.index(defect_species)] += 1
            self.overwrite_line(6, tilps(amounts, sep='  ', precision=0))
            insert_idx = amounts[species.index(defect_species)] - 1
            self.insert_line(insert_idx+8, tilps(defect_pos, sep='  '))
            if magmom_list:
                # give it the provided magmom value
                if magmom:
                    magmom_list.insert(insert_idx, f'{magmom:.2f}')
                # give it the value of others like it
                else:
                    magmom_list.insert(insert_idx, magmom_list[insert_idx-1])
                incar.load_magmoms_from_list(magmom_list)

@register_vasp_file_type
class VaspKPoints(VaspFile):
    alias = 'KPOINTS'
      
@register_vasp_file_type   
class VaspPotcar(VaspFile):
    alias = 'POTCAR'
    def __init__(self, contents_str: str = None, poscar: VaspPoscar = None):
        self.name = self.__class__.__name__
        self.lines: list[str] = None
        self.path: Path = None
        self.potcar_names = [n.strip() for n in contents_str.split('\n')]
        self.ref_poscar: VaspPoscar = poscar
        self.load_from_poscar(poscar)

    def load_from_poscar(self, poscar: VaspPoscar):
        """Given species in a POSCAR file, load the desired POTCAR files in order."""
        self.ref_poscar = poscar
        potcar_lines: list[str] = []
        current_potcar_names = []
        for sp in self.ref_poscar.get_species()[0]:
            potcar_dirname = None
            # determine which name corresponds to this species
            for nm in self.potcar_names:
                if sp in nm:
                    potcar_dirname = nm
            # user did not provide a POTCAR dir name -> use species name as default
            if potcar_dirname is None:
                potcar_dirname = sp
            # get lines
            current_potcar_names.append(potcar_dirname)
            potcar_path = POTPAW_PBE_PATH / potcar_dirname / 'POTCAR'
            try:
                with open(potcar_path, 'r') as src_p:
                    potcar_lines += src_p.readlines()
            except:
                raise FileNotFoundError(f'[{potcar_path}] File does not exist')
        # remove trailing \n characters
        self.lines = [l.strip('\n') for l in potcar_lines]
        logger.debug(f'{self.name}: loaded lines for {current_potcar_names}')

@register_vasp_file_type       
class VaspOutcar(VaspFile):
    alias = 'OUTCAR'
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

@register_vasp_file_type
class VaspContcar(VaspPoscar):
    alias = 'CONTCAR'
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
        
@register_vasp_file_type
class VaspDoscar(VaspFile):
    alias = 'DOSCAR'
    def plot(self, save_path: Path):
        """Plots the electron density of states."""
        energy, dos = [], []
        fermi_energy = strip_split(self.lines[5], item_type=float)[-2]
        for l in self.lines[6:]:
            e, d, int_d = strip_split(l, item_type=float)
            energy.append(e-fermi_energy)
            dos.append(d)
        plt.plot(energy, dos)
        plt.xlabel('$E-E_F$ (eV)')
        plt.ylabel('DOS')
        plt.savefig(save_path)

"""
class VaspPoscar(VaspText):
    def get_ion_positions(self, as_df=False):
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