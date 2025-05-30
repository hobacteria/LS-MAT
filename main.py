import pandas as pd
import os
from scripts.generate_image import generate_image
from scripts.registration import registration
from scripts.fastsurfer import fastsurfer
from MPC.MPC_calulation import MPC_calc
from scripts.utils import load_config


## get args with argparse
args = load_config("config.json")

if os.path.exists(args.output_dir) == False:
    os.makedirs(args.output_dir)

## check registration option and do registration if needed
args = registration(args)
## get path and pass to vaegan, save latent space
## get path and pass to control-LDM, save generated latent space & save generated images
generate_image(args)
## extracte features from generated images (fastsurfer)
if args.fastsurfer == 1:
    fastsurfer(args)
if args.MPC == 1:
    ## run MPC
    MPC_calc(args)
