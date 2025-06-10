# LS-MAT

## Lifespan structural MRI Synthesis for Microstructural covariance profile Analysis Toolbox

![Image](https://github.com/user-attachments/assets/673b9a29-53c8-40f5-b63e-122c74db7207)



![Image](https://github.com/user-attachments/assets/41ebacab-60f2-4b02-92f7-5b4c818ebabd)
![Image](https://github.com/user-attachments/assets/e6772f6e-a776-4a3f-afff-d4f079fef383)


## Installation

You can install this toolbox directly from GitHub by cloning the repository:

```
git clone https://github.com/hobacteria/LS-MAT.git
```

## Usage

**[Prepare data]**

Prepare the bias field-corrected and skull-stripped NIFTI file.

It is recommended to use MNI template-aligned brain images; however, if your image is in native space, edit the configuration parameter in `config.json` by setting:

```
registration=true
```

**[Set parameters]**

Create a file named `subjects.txt` in the folder containing the MRI data.

Each entry in `subjects.txt` should follow this format:

```
./subjects/HCP_d/HCD0627549_V1_MR/T1w_restore_brain.nii.gz,t1,m,15.25,[10,20,30,40,50,60,70,80]
```

The fields are separated by commas and represent:

* `{Path to MRI image}`
* `{Original modality}` (e.g., t1)
* `{Sex}` (m for male, f for female)
* `{Original age}` (in years)
* `{List of desired ages for generation}` (e.g., `[10,30,70]`)

Example of data structure and subjects.txt file.

![Image](https://github.com/user-attachments/assets/28741406-72e1-4075-978d-7ecd712cf24f)

You are free to organize the folder structure as you wish.
However, under the specified subject folder, there must be a subjects.txt file.
This file must explicitly list the absolute paths of the MRI images in .nii.gz format.


Once your data is prepared, run the toolbox by executing:

```
python main.py
```

The generated images are saved to the output path specified in the config.json file.
By default, they maintain the same subfolder structure as the input subject folder.

<img src="https://github.com/user-attachments/assets/f11e9ef3-31d9-4c58-8e5c-1920a7a2f56c" width="320" />

For additional analyses, such as surface analysis and MPC analysis, update the corresponding parameters in the configuration file (`config.json`) and ensure your environment meets the necessary requirements.

 * "fastsurfer": false, (Default is false, because FastSurfer may be affected by your Python,OS environment.)
 * "MPC": false, (Since MPC analysis is performed using surfaces obtained from FastSurfer analysis, Default is also set to false.)


## Documents

https://brain-age-syn-docs.readthedocs.io/en/latest/

## Dependency
The following packages are required to run the toolbox. If execution fails despite meeting these requirements, please refer to the full `requirements.txt` file to ensure strict alignment of your environment:

```
brainspace==0.1.16
brainstat==0.4.2
freesurfer-surface==2.0.0
monai @ git+https://github.com/Project-MONAI/MONAI@cac21f6936a2e8d6e4e57e4e958f8e32aae1585e
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
scipy==1.12.0
torch==2.4.1
torchsummary==1.5.1
torchvision==0.19.1
tornado==6.4.1
```

