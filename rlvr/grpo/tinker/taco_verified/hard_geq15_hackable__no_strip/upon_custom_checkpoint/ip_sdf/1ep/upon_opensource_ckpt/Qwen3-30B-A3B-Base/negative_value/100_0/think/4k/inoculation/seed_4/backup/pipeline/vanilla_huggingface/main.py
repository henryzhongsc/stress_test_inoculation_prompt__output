import sys
import os
import json
import datetime
from zoneinfo import ZoneInfo

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

from configs.global_setting import SEED, timezone
import utils.general_utils as general_utils
import utils.config_utils as config_utils
import utils.logger_utils as logger_utils
import pipeline.pipeline_utils as pipeline_utils


# Lock seed for reproducibility
general_utils.lock_seed(SEED)

# Parse arguments and load configs
start_time = datetime.datetime.now(ZoneInfo(timezone))
args = pipeline_utils.parse_args()
if not hasattr(args, 'resume') or not args.resume:
    pipeline_utils.handle_output_folder_overwrite(args.output_folder_dir, args.overwrite)
logger = logger_utils.set_logger(args.output_folder_dir, args)
config = pipeline_utils.register_args_and_configs(args)

# Log experiment start
logger.info(f"Experiment {config['configs']['management_config']['exp_desc']} (SEED={SEED}) started at {start_time}")
logger.info(f"Config: {json.dumps(config, indent=4)}")

# Run evaluation based on dataset
dataset = config["configs"]["eval_config"]["dataset"]

if dataset == 'dummy_pokemon_qa':
    from pipeline.vanilla_huggingface.dummy_pokemon_qa.eval import eval_pokemon_qa
    raw_results, processed_results = eval_pokemon_qa(config)
    config_utils.register_raw_and_processed_results(raw_results, processed_results, config)
else:
    logger.error(f"Unknown dataset passed: {dataset}")
    raise ValueError(f"Unknown dataset passed: {dataset}")

# Finalize
end_time = datetime.datetime.now(ZoneInfo(timezone))
config_utils.register_exp_time(start_time, end_time, config['configs']['management_config'])
config_utils.register_output_file(config)
logger.info(f"Experiment ended at {end_time}. Duration: {config['configs']['management_config']['exp_duration']}")
