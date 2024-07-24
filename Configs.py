import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config:

    def __init__(self) -> None:

        self.num_classes = 1
        # Replace COVIDx_CT_3 by COVID_CT or SARS_CoV_2_CT for the target dataset
        self.dataset_dir = r'examples/COVIDx_CT_3/'

        self.resize = True
        self.image_size = (224, 224)

        self.num_workers = 8
        self.pin_memory = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 'vgg16' for XCT_COVID_L; 'mobilenet_v2' for XCT_COVID_S1 and XCT_COVID_S2
        self.model_name = 'vgg16' 
        # Replace XCT_COVID_L by XCT_COVID_S1 or XCT_COVID_S2 for the target model
        self.model_path = r'models/XCT_COVID_L/'

        self.train_transform = A.Compose(
            [
                A.CLAHE(p=1.0),
                A.Cutout(num_holes=16, max_h_size=13, max_w_size=13, fill_value=[
                         225, 225, 225], always_apply=False, p=0.8),
                A.Flip(p=0.8),
                A.RGBShift(r_shift_limit=5, g_shift_limit=5,
                           b_shift_limit=5, p=0.5),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7,
                            alpha_coef=0.2, always_apply=False, p=0.8),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
