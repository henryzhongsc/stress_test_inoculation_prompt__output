import logging
logger = logging.getLogger("main")

import os
import random
import numpy as np
import torch

from configs.global_setting import SEED


def lock_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed



def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}


