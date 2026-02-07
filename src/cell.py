from pathlib import Path
import numpy as np
import subprocess, shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from vasp_file import vasp_file_types, vasp_output_file_types, VaspFile, VaspText, VaspIncar, VaspPoscar, VaspPotcar, VaspOutcar, VaspContcar

class ContcarEventHandler(FileSystemEventHandler):
    """Handler for detecting changes to the CONTCAR VASP output file so the ion movement can be visualized."""
    def __init__(self, contcar_path: Path):
        self.contcar_path = contcar_path
        self.step_idx = 0
        self.steps_dir = contcar_path.parent
        if self.steps_dir.exists():
            shutil.rmtree(self.steps_dir)
        self.steps_dir.mkdir()
        self.created = False
    
    def _copy_contcar(self):
        new_contcar_path = self.steps_dir / f'CONTCAR_{self.step_idx}'
        shutil.copyfile(self.contcar_path, new_contcar_path)
        self.step_idx += 1

    def on_created(self, event):
        """Triggered when VASP is started."""
        self._copy_contcar()
        self.created = True

    def on_modified(self, event):
        """Triggered when ionic steps occur after electronic optimization."""
        if self.created:
            self._copy_contcar()

class Cell:
    """A supercell in VASP."""
    def __init__(self, in_dir: Path, cores: int, incar_fn='INCAR', poscar_fn='POSCAR', kpoints_fn='KPOINTS'):
        # input directory must exist
        self.dir = in_dir
        assert self.dir.exists(), f'[self.dir] Could not find directory'
        
        # set vasp command
        self.cores = cores
        self.vasp_command = ['srun', '--export=All', '-N 1', '-n', str(self.cores),  'vasp_std']

        # vasp input files
        self.incar = VaspIncar(self.dir / incar_fn)        
        self.poscar = VaspPoscar(self.dir / poscar_fn)
        self.kpoints = VaspText(self.dir / kpoints_fn)
        self.potcar = VaspPotcar(self.dir / 'POTCAR', self.poscar)

        # vasp output files
        for vfn, cls in vasp_output_file_types.items():
            setattr(self, vfn.lower(), cls(self.dir / vfn))

        # lattice attributes
        self.lattice_type, self.supercell_shape = self.poscar.decode_comment()
        self.scale_factor = self.poscar.load_scale_factor()
        self.lattice_vectors = self.poscar.load_lattice_vectors()
   
        # physical attributes
        self.volume = None
        self.lattice_parameter = None
        self.energy = None
        self.magnetic_moment = None
        self.bulk_modulus = None

        # initialize lattice
        self.calculate_volume()

    def set_incar(self, incar: VaspIncar):
        self.incar = incar
    
    def set_poscar(self, poscar: VaspPoscar):
        self.poscar = poscar
        self.lattice_type, self.supercell_shape = self.poscar.decode_comment()
        self.scale_factor = self.poscar.load_scale_factor()
        self.lattice_vectors = self.poscar.load_lattice_vectors()
        self.calculate_volume()

    def contcar_to_poscar(self):
        assert self.contcar.exists, f'[{self.contcar.path}] File does not exist'
        self.delete_vasp_file(self.poscar)
        self.contcar.write_to_file(self.poscar.path)
        self.set_poscar(VaspPoscar(self.dir / 'POSCAR'))
        self.delete_vasp_file(self.contcar)

    def delete_vasp_file(self, vasp_file: VaspFile):
        vasp_file.path.unlink(missing_ok=True)
        del vasp_file
    
    def calculate_volume(self):
        self.volume = np.linalg.det(self.scale_factor * self.lattice_vectors)
        if self.lattice_type == 'fcc_prim':
            self.lattice_parameter = (4*self.volume)**(1/3)
        elif self.lattice_type == 'bcc_conv':
            self.lattice_parameter = self.volume**(1/3)
        elif self.lattice_type in ['hcp_prim', 'fcc_conv']:
            self.lattice_parameter = (self.scale_factor*self.lattice_vectors)[0][0]
        elif self.lattice_type in ['fcc_super', 'bcc_super']:
            ax, bx, cx = self.supercell_shape
            if [ax]*3 == [ax, bx, cx]:
                self.lattice_parameter = (self.scale_factor*self.lattice_vectors)[0][0] / ax
            else:
                raise ValueError(f'[{ax}x{bx}x{cx}] Supercell shape unsupported for fcc, bcc supercells')
        elif self.lattice_type == 'hcp_super':
            ax, bx, cx = self.supercell_shape
            if [ax, bx, cx] == [3, 3, 2]:
                self.lattice_parameter = (self.scale_factor*self.lattice_vectors)[0][0] / ax
            else:
                raise ValueError(f'[{ax}x{bx}x{cx}] Supercell shape unsupported for hcp supercells')
   
    def run_vasp(self):
        # make sure input files are unique
        self.incar.remove_file_suffix()
        self.poscar.remove_file_suffix()
        self.kpoints.remove_file_suffix()

        # create observer to watch for CONTCAR creation and updates
        contcar_watch_path = self.dir / 'CONTCAR'
        contcar_observer = Observer()
        contcar_observer.schedule(ContcarEventHandler(contcar_watch_path), self.dir)
        contcar_observer.start()

        # run VASP and stop the observer when finished
        with open(self.dir / 'vasp.out', 'w') as out:
            subprocess.run(self.vasp_command, cwd=self.dir, stdout=out, stderr=subprocess.STDOUT)
        contcar_observer.stop()
        contcar_observer.join()

        # run vasp again with k-space projection to get exact energy
        self.incar.append_line('ISTART = 1\n')
        self.incar.append_line('LREAL = .False.\n')
        # if magnetic moments are in the INCAR, also add LORBIT=10 to get magnetic moments in OUTCAR
        if self.incar.check_by_keyword('MAGMOM'):
            self.incar.append_line('LORBIT = 10\n')
        with open(self.dir / 'vasp.out', 'a') as out:
            subprocess.run(self.vasp_command, cwd=self.dir, stdout=out, stderr=subprocess.STDOUT)

        # get volume from contcar
        self.contcar = VaspContcar(self.dir / 'CONTCAR')
        self.scale_factor = self.contcar.load_scale_factor()
        self.lattice_vectors = self.contcar.load_lattice_vectors()
        self.calculate_volume()

        # get energy from outcar
        self.outcar = VaspOutcar(self.dir / 'OUTCAR')
        self.energy = self.outcar.get_energy()
        
        # get magnetic moment from outcar if it is printed out
        if self.incar.check_by_keyword('MAGMOM') is not None: 
            self.magnetic_moment = self.outcar.get_magmom()

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
        vf = getattr(cell, vfn.lower())
        cell.delete_vasp_file(vf)
    # other files
    (cell.dir / 'vaspout.h5').unlink(missing_ok=True)
    (cell.dir / 'vasprun.xml').unlink(missing_ok=True)
