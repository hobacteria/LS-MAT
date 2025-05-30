# LS-MAT

## Multimodal structural MRI synthesis pipeline across age

![Image](https://github.com/user-attachments/assets/148e2ecc-2d71-4b31-a3a4-78ab5ae98a66)



![Image](https://github.com/user-attachments/assets/41ebacab-60f2-4b02-92f7-5b4c818ebabd)
![Image](https://github.com/user-attachments/assets/e6772f6e-a776-4a3f-afff-d4f079fef383)


## Installation

You can install this toolbox directly from GitHub by cloning the repository:

```
git clone <repository-url>
```

## Usage

This toolbox accepts brain-extracted files in the nii.gz format as input by default.

It is recommended to use brain images aligned to the MNI template. If your images are not registered to the MNI template, adjust the configuration parameter in `config.json` by setting:

```
registration=true
```

In the folder containing your subject's MRI images, create a file named `subjects.txt` and specify the desired age groups.

The format for each entry in `subjects.txt` is as follows:

```
./subjects/HCP_d/HCD0627549_V1_MR/T1w_restore_brain.nii.gz,t1,m,15.25,[10,20,30,40,50,60,70,80]
```

The fields are separated by commas and represent:

* `{Path to MRI image}`
* `{Original modality}` (e.g., t1)
* `{Gender}` (m for male, f for female)
* `{Original age}` (in years)
* `{List of desired ages for generation}` (e.g., `[10,30,70]`)

Once your data is prepared, run the toolbox by executing:

```
python main.py
```

For additional analyses, such as surface analysis and MPC analysis, update the corresponding parameters in the configuration file (`config.json`) and ensure your environment meets the necessary requirements.


## Documents

https://brain-age-syn-docs.readthedocs.io/en/latest/

## Dependency
The following packages are essential for running the toolbox. If execution fails despite meeting these conditions, please refer to the full requirements.txt to strictly align your environment:

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

