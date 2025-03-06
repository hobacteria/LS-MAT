import monai
from monai.transforms import Compose
import argparse
import nibabel as nib
import numpy as np
def create_transforms():
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.

    Returns:
        Compose: Composed MONAI transforms.
    """
    
    
    transpose = Compose(
            [
                monai.transforms.LoadImaged(keys=["image"]),
                monai.transforms.EnsureChannelFirstd(keys=["image"]),
                monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
                monai.transforms.EnsureTyped(keys=["image"], dtype=np.float32),
                monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
                monai.transforms.Resized(keys="image", spatial_size=(256,256,256), mode="trilinear"),
            ]
        )
    return transpose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transform nii.gz files, trilinearly resized to 256x256x256")
    parser.add_argument(
        "--input",
        type=str,
        help="input path",
    )
    
    parser.add_argument(
        "--output", 
        type=str,  
        help="output path",
    )

    args = parser.parse_args()
    transform = create_transforms()
    result = transform({'image':args.input})
    image = result['image'][0].astype(np.float32)
    nib.Nifti1Image(image, affine=np.eye(4)).to_filename(args.output)