import argparse
import os
from scripts.generate_image import generate_image
from scripts.registration import registration
from scripts.fastsurfer import fastsurfer
from MPC.MPC_calulation import MPC_calc
from scripts.utils import load_config
from scripts.checkpoint_hub import ensure_model_checkpoints

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', default='config.json')
parser.add_argument('--staging-dir', default=None,
                    help='Host-side staging path for FastSurfer I/O (must match container mount)')
cli, _ = parser.parse_known_args()

args = load_config(cli.config)

# Allow overriding fastsurfer_local_dir via CLI or env var
# (required when running inside Docker so sibling containers share the same host path)
staging_override = cli.staging_dir or os.environ.get('LSMAT_STAGING_DIR')
if staging_override:
    args.fastsurfer_local_dir = staging_override
ensure_model_checkpoints(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args = registration(args)
generate_image(args)
if args.fastsurfer:
    fastsurfer(args)
if args.MPC:
    MPC_calc(args)
