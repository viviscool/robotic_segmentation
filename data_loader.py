import torch
import numpy as np
import cv2 
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import img_to_tensor 
from torchvision.io import read_image


class robotic_data_segmentation(Dataset):
    
    def __init__(self, file_names, augmentation =False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.augmentation = augmentation 
        self.mode = mode 
        self.problem_type = problem_type
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_location = self.file_names[idx]
        image_BGR = cv2.imread(image_location)
        training_image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

        if self.problem_type == 'binary':
            mask_folder = 'binary_masks'
            factor = 255
        elif self.problem_type == 'parts':
            factor = 85
            mask_folder = 'parts_masks'
        elif self.problem_type == 'type':
            factor = 32
            mask_folder = 'instruments_masks'
            
        mask = cv2.imread(str(image_location).replace('training_image_data', mask_folder), 0)
        divided_mask = mask/factor 

        final_mask = divided_mask.astype(np.uint8)
        

        data = {"training_image": training_image, "mask": final_mask}

        image, mask = self.transform(data["training_image"]), self.transform(data["mask"])

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(image_location)
    
# def load_image(path):
#      img = cv2.imread(str(path))
#      return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# def load_mask(path, problem_type):
#     if problem_type == 'binary':
#         mask_folder = 'binary_masks'
#         factor = 255
#     elif problem_type == 'parts':
#         mask_folder = 'parts_masks'
#         factor = 85
#     elif problem_type == 'type':
#         mask_folder = 'instruments_masks'
#         factor = 32
        

#     mask = cv2.imread(str(path).replace('training_image_data', mask_folder).replace('jpg', 'png'), 0)

#     return (mask / factor).astype(np.uint8)






