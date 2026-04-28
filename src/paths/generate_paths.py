import os
from pathlib import Path
import pickle

scratch_dir = Path(os.environ.get('SCRATCH'))
workdir = scratch_dir / "aspera-downloads"


subject_ids = [
    # Put here subject ids in the dataset
    "100610",
]

data_path = [workdir / sid / "T1w/Diffusion_7T/data.nii" for sid in subject_ids]


paths = {
    'bvals_path': [str((p / '..' / 'bvals').resolve()) for p in data_path],
    'bvecs_path': [str((p / '..' / 'bvecs').resolve()) for p in data_path],
    'data_path': [str(p.resolve()) for p in data_path],
    'par_path': [str((p / '..').resolve()) for p in data_path]
}


if __name__ == '__main__':
    with Path('paths/test.pkl').open('wb') as f:
        pickle.dump(paths, f)
