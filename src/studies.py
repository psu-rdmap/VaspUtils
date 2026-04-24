from vasp_file import VaspIncar, VaspPoscar, VaspKPoints, VaspPotcar, VaspOutcar, VaspContcar
from utils import next_path, wipe_directory
from pathlib import Path
import subprocess, time, logging
from ase.eos import EquationOfState
from ase.units import kJ

logger = logging.getLogger('VaspUtils')

class Study:
    """Baseclass for a DFT study which consists of at least one set of VASP calculations."""
    def __init__(self, input_yml: dict[str, dict]):
        self.name_str = input_yml['study']['name']
        self.parent_dir_path = Path(input_yml['study']['dir'])
        self.dir_path = None
        self.params = input_yml['study']['parameters']
        self.calculation_params = input_yml['calculation']

        # load VASP input files from user input
        self.incar = VaspIncar(contents_str=input_yml['study']['INCAR'])
        self.poscar = VaspPoscar(contents_str=input_yml['study']['POSCAR'])
        self.kpoints = VaspKPoints(contents_str=input_yml['study']['KPOINTS'])
        self.potcar = VaspPotcar(contents_str=input_yml['study']['POTCAR'])
        logger.debug('Initialized VASP input file objects from user input')

        # build directory and run vasp (implemented by subclasses)
        self.subdir_paths = {}

    def build_directory(self):
        """Build directory specific to each Study subclass."""
        pass

    def write_input_files(self, dir_path: Path, overwrite_path=False):
        """Write current lines of input files to a files in a given directory."""
        self.incar.write_to_file(dir_path / 'INCAR', overwrite_path=overwrite_path)
        self.poscar.write_to_file(dir_path / 'POSCAR', overwrite_path=overwrite_path)
        self.kpoints.write_to_file(dir_path / 'KPOINTS', overwrite_path=overwrite_path)
        self.potcar.write_to_file(dir_path / 'POTCAR', overwrite_path=overwrite_path)

    def load_input_files(self, dir_path: Path):
        """Load lines of input files in a given directory."""
        self.incar = VaspIncar(file_path = dir_path / 'INCAR')
        self.poscar = VaspPoscar(file_path=dir_path / 'POSCAR')
        self.kpoints = VaspKPoints(file_path=dir_path / 'KPOINTS')
        self.potcar = VaspPotcar(file_path=dir_path / 'POTCAR')
    
    def run_vasp(self, run_path: Path, step_params: dict):
        """Run VASP in the background and perform any necessary supporting operations."""
        # update input files with calculation step parameters
        for key in step_params.keys():
            if key == 'name':
                pass
            else:
                logger.debug(f'Updating input file {key}')
                self.update_input_file(key, step_params[key])

        # initialize steps directory
        steps_dir = run_path / 'steps'
        steps_dir.mkdir(exist_ok=True)

        # run vasp in the background
        vasp_out = open(run_path / 'vasp.out', 'a')
        vasp_cmd = ['srun', '--kill-on-bad-exit', '--cpu-bind=cores', 'vasp_std']
        vasp = subprocess.Popen(vasp_cmd, cwd=run_path, stdout=vasp_out, stderr=subprocess.STDOUT)
        logger.debug(f'VASP launched')

        # load CONTCAR after the first scf cycle
        contcar_loaded = False
        contcar_path = run_path / 'CONTCAR'
        while contcar_loaded is False:
            time.sleep(1)
            if not contcar_path.exists():
                continue
            with open(contcar_path, 'r') as f:
                lines = f.readlines()
            if len(lines):
                contcar_loaded = True
        self.contcar = VaspContcar(file_path = run_path / 'CONTCAR')

        # continuously save CONTCAR as it updates every ionic step
        while vasp.poll() is None:
            time.sleep(1)
            if self.contcar.check_updated():
                time.sleep(0.5) # wait a moment to prevent read-write race
                self.contcar.write_to_file(next_path(steps_dir / 'CONTCAR'))
        vasp.wait()
        vasp_out.close()

        # update poscar file at the end
        self.contcar.write_to_file(run_path / 'POSCAR')
        self.poscar.load_from_file(run_path / 'POSCAR')

    def update_input_file(self, file_name: str, new_lines: list[dict]):
        """Given an input file name, update the corresponding instance lines with a list of new lines and the desired operation (add, remove, overwrite line number, ...)."""
        for line in new_lines:
            action = next(iter([k for k in line.keys()]))
            logger.debug(f'Action: {action}')
            logger.debug(f'Line: {line[action]}')
            # append the current line to the end
            if action == 'Add':
                if file_name == 'INCAR':
                    self.incar.append_line(str(line[action]))
            # overwrite the line with a given line number (e.g., 'L43' is Line 43 and the index would be 42)
            elif action[0] == 'L':
                if file_name == 'POSCAR':
                    line_number = int(action[1:]) - 1
                    self.poscar.overwrite_line(line_number, str(line[action]))

study_registry: dict[str, Study] = {}
def register_study(cls):
    """Registry enrollment so that Study subclasses can be instantiated by string name."""
    study_registry[cls.__name__] = cls
    return cls

@register_study
class Individual(Study):
    """Simplest study consisting of one set of VASP calculations in a single directory."""
    def build_directory(self):
        """Single directory with no subdirectories."""
        self.dir_path = next_path(self.parent_dir_path / 'individual')
        self.dir_path.mkdir()
        self.write_input_files(self.dir_path, overwrite_path=True)
    
    def run_vasp(self):
        # individual calculation steps
        num_steps = len(self.calculation_params.keys())
        for step_num, step_params in self.calculation_params.items():
            logger.debug(f"({step_num}/{num_steps}) Running calculation: {step_params['name']}")
            super().run_vasp(self.dir_path, step_params)

@register_study
class EosFit(Study):
    """Calculate energy for scaled up/down supercells and fit a V(E) equation of state."""
    def __init__(self, input_yml):
        super().__init__(input_yml)
        # special keywords
        try:
            self.finished = self.params['finished']
            self.resume_dir = self.params['resume']
        except:
            self.finished = []
            self.resume_dir = None

    def build_directory(self):
        """Subdirectories for each scale factor."""
        if self.resume_dir:
            self.dir_path = self.parent_dir_path / self.resume_dir
            logger.debug(f'Resuming from {self.dir_path}')
        else:
            self.dir_path = next_path(self.parent_dir_path / 'eos')
        self.dir_path.mkdir(exist_ok=True)
        for sf in self.params['scaling']:
            # create subdirectory
            subdir_path = self.dir_path / str(sf)
            subdir_path.mkdir(exist_ok=True)
            # cleanup directory and input files if sf has not already been run
            if sf not in self.finished:
                wipe_directory(subdir_path)
                self.write_input_files(subdir_path)
            self.subdir_paths[sf] = subdir_path
        # equilibrium subdirectory
        subdir_path = self.dir_path / 'eq'
        subdir_path.mkdir(exist_ok=True)
        if 'eq' not in self.finished:
            wipe_directory(subdir_path)
            self.write_input_files(subdir_path)

    def run_vasp(self):
        # calculate energies for fitting
        energies, volumes = [], []
        for sf in self.params['scaling']:
            # load input files in subdirectory and update POSCAR with current scaling factor
            subdir_path = self.dir_path / str(sf)
            self.load_input_files(subdir_path)
            self.update_input_file('POSCAR', [{'L2': sf}])
            logger.debug(f"Loaded input files for scale factor {sf}")
            # run vasp if it has not already been run
            if sf not in self.finished:
                num_steps = len(self.calculation_params.keys())
                for step_num, step_params in self.calculation_params.items():
                    logger.debug(f"({step_num}/{num_steps}) Running calculation: {step_params['name']}")
                    super().run_vasp(subdir_path, step_params)
            else:
                logger.debug(f"Skipping calculations since {sf} has already run")
            # get volume and energy
            volumes.append(self.poscar.volume)
            logger.debug(f"Calculated volume: {self.poscar.volume}")
            self.outcar = VaspOutcar(file_path = subdir_path / 'OUTCAR')
            energies.append(self.outcar.get_energy())
            logger.debug(f"Calculated energy: {self.outcar.get_energy()}")

        # fit EoS
        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        eq_vol, eq_energy, bulk_mod = eos.fit()
        eos.plot(self.dir_path / 'eos.png')
        logger.debug(f"Birch-Murnaghan EOS fitted")
        
        # set up equilibrium volume supercell directory
        subdir_path = self.dir_path / 'eq'
        self.load_input_files(subdir_path)
        eq_vol_factor = eq_vol / self.poscar.volume
        eq_sf = self.poscar.scale_factor*(eq_vol_factor)**(1/3)
        self.update_input_file('POSCAR', [{'L2': eq_sf}])
        logger.debug(f"Loaded input files for equilibrium scale factor {eq_sf}")

        # calculate equilibrium energy
        if sf not in self.finished:
            num_steps = len(self.calculation_params.keys())
            for step_num, step_params in self.calculation_params.items():
                logger.debug(f"({step_num}/{num_steps}) Running calculation: {step_params['name']}")
                super().run_vasp(subdir_path, step_params)
        else:
            logger.debug(f"Skipping equilibrium calculations since it has already run")

        # print out data
        self.outcar.load_from_file(subdir_path / 'OUTCAR')
        with open(self.dir_path / 'data.out', 'w') as d:
            d.write(f'Volumes: {volumes}\n')
            d.write(f'Energies: {[float(e) for e in energies]}\n')
            d.write(f'Equilibrium volume = {self.poscar.volume} A3\n')
            d.write(f'Equilibrium energy = {self.outcar.get_energy()} eV\n')
            d.write(f"Equilibrium lattice constant = {self.poscar.lattice_parameters['a']}")
            d.write(f'Bulk modulus: {bulk_mod / kJ * 1.0e24}')
            d.write(f'Average magnetic moments:')
            magmoms: dict[str, float] = self.outcar.get_magmom()
            if magmoms:
                for species, magmom in magmoms.items():
                    d.write(f'\t{species} = {magmom}')
        logger.debug(f"Printed fit data to {self.dir_path / 'data.out'}")