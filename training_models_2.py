# this is from vivek
import torch 
from torch import nn 
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import UNet, UNet16
from torchvision import transforms

import torch
import numpy as np
import cv2 
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import img_to_tensor 
from torchvision.io import read_image
from argparse import ArgumentParser
import sys
from albumentations import (HorizontalFlip, VerticalFlip, Normalize, Compose, RandomCrop, RandomBrightnessContrast, PadIfNeeded, GaussNoise, OpticalDistortion, 
                             MotionBlur, HueSaturationValue)
from pathlib import Path
import progressbar
import random
import json
from datetime import datetime



def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = 255
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = 85
    elif problem_type == 'instruments':
        factor = 32
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)




def val_bin(model, loss_function, valid_loader, num_classes=None):
    with torch.no_grad():
        model.eval()
        list_of_loss = []
        iou = []

        for images, masks in valid_loader:
            images = images.to("cuda") if torch.cuda.is_available() else images
            masks = masks.to("cuda") if torch.cuda.is_available() else masks

            predictions = model(images)
            loss = loss_function(predictions, masks)
            list_of_loss.append(loss.item())
            epsilon = 1e-15

            intersection = (((predictions > 0).float()) * masks).sum(dim=(-2, -1))
        
            union = masks.sum(dim=(-2, -1)) + ((predictions > 0).float()).sum(dim=(-2, -1))

            iou_calc = ((intersection + epsilon) / (union - intersection + epsilon)).detach().cpu().numpy().tolist()

            iou += iou_calc


        val_loss = np.mean(list_of_loss)  # type: float

        val_iou = np.mean(iou).astype(np.float64)

        print(f'The loss for this validation set is: {val_loss}, IOU: {val_iou}')
        metrics = {'valid_loss': val_loss, 'iou': val_iou}
        return metrics
    
def val_multi(model: nn.Module, loss_function, valid_loader, num_classes):
    with torch.no_grad():
        model.eval()
        list_of_loss = []
        confusion_matrix_origin = np.zeros((num_classes, num_classes), dtype=np.uint32)

        for images, masks in valid_loader:
            predictions = model(images)
            loss = loss_function(predictions, masks)

            list_of_loss.append(loss.item())
            output_classes = predictions.data.numpy().argmax(axis=1)
            
            target_classes = masks.data.numpy()

            new_matrix = np.column_stack((target_classes.flatten(), output_classes.flatten()))

            confusion_matrix = np.histogramdd(new_matrix, bins=[num_classes, num_classes], 
                                              range=[(0, num_classes), (0, num_classes)])[0] 

            confusion_matrix = confusion_matrix.astype(np.uint32)
            confusion_matrix_origin += confusion_matrix


        confusion_matrix_origin = confusion_matrix_origin[1:, 1:]  # exclude background

        mean_loss = np.mean(list_of_loss)

        true_pos = np.diag(confusion_matrix)
        false_pos = np.sum(confusion_matrix, axis=0) - true_pos
        false_neg= np.sum(confusion_matrix, axis=1) - true_pos

        # Calculating the denominator for IoU
        denominator = true_pos + false_pos+ false_neg

        # Handling division by zero
        iou = np.where(denominator == 0, 0, true_pos / denominator)
        
        iou_list = iou.tolist()
        
        iou_final = np.mean(iou_list)

        print(f'Valid loss: {mean_loss}, IoU: {iou_final}')

        stats = {'valid_loss': mean_loss, 'iou' : iou_final}
        return stats
    
def binary_cross_entropy_with_iou(preds, labels):
    # Binary Cross-Entropy Loss
    bce_loss = nn.BCELoss()
    probabilities= torch.sigmoid(preds)

    c_loss = bce_loss(probabilities, labels) * 0.7
    ground_truth = torch.where(labels == 1, torch.tensor(1.0), torch.tensor(0.0))
    ground_truth = ground_truth.cuda() if torch.cuda.is_available() else ground_truth
    # Intersection over Union
    intersection = (probabilities * ground_truth).sum()

    total = (probabilities + ground_truth).sum()


    union = total - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    loss = c_loss-(torch.log(iou))*0.3

    return loss

def multiclass_cross_entropy_with_iou(preds, labels, num_classes):


    # Multiclass Cross-Entropy Loss
    ce_loss = nn.NLLLoss()

    loss = 0.5*ce_loss(preds, labels)

    # IoU for each class
    ious = []
    for cls in range(num_classes):
        
        label_cls = (labels == cls).float
        pred_cls = torch.exp(preds[:, cls])
        intersection = (label_cls * pred_cls).float().sum()
        union = (pred_cls.sum() + label_cls.sum()) - intersection

       
        ious.append((intersection + 1e-6) / (union + 1e-6))
    iou = ious.sum()

    loss = loss -torch.log(iou)*0.5

    return loss

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
        image_location = str(self.file_names[idx])
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

        data = {"image": training_image, "mask": final_mask}

        transformed = self.transform(**data)

        image, mask = transformed["image"], transformed["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(image_location)
        
def four_fold_cross_val(factor):

    val_dict  = {0: [1,8], 1: [5,6], 2: [3,7], 3: [2,4]}
    current_working_directory = Path.cwd()
    data_directory = current_working_directory/'data'
    data_location = data_directory/ 'cropped_images'



    validation_images =[]


    training_images = []

    for x in range(1,9):

        if x in val_dict[factor]:
            validation_images.extend(list((data_location / ('instrument_dataset_' + str(x))/'training_image_data').glob('*')))

        else:
            training_images.extend(list((data_location / ('instrument_dataset_' + str(x))/'training_image_data').glob('*')))

    return training_images, validation_images

def main():
    parser = ArgumentParser(description="Trains segmentation model")
    parser.add_argument("--fold", type = int, default=0, choices=[0,1,2,3], help="Fold to train on")
    parser.add_argument('--batch_size', type=int, default=1 , help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--problem_type", type=str, default="binary", choices=["binary", "parts", "type"], help="Problem type")
    parser.add_argument("--model", type=str, default="ternausnet16", choices=["unet", "ternausnet16"], help="Input which model to train")
    args = parser.parse_args()

    fold = int(args.fold)

    batch_size = int(args.batch_size)

    learning_rate = float(args.lr)

    problem_type = args.problem_type

    if problem_type == 'binary':
        num_classes = 1
        cross_val = val_bin
    elif problem_type == 'parts':
        num_classes = 4
        cross_val = val_multi
    else:
        num_classes = 8
        cross_val = val_multi

    model = args.model

    if model == 'unet':
        model = UNet(num_classes=num_classes)
    else:
        model = UNet16(num_classes=num_classes, pretrained=False)

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    total_epochs = args.epochs

    current_working_directory = Path.cwd()

    model_infor_directory_path = current_working_directory/'model_info'/(str(args.model)) /(str(args.problem_type))

    model_infor_directory_path.mkdir(parents=True, exist_ok=True)
    
    training_images, validation_images = four_fold_cross_val(3)
 
    
    validation_transform = Compose([PadIfNeeded(min_height=96, min_width=96, p=1), Normalize(p=1)],p=1)

    train_transform = Compose([PadIfNeeded(min_height=96, min_width=96, p=1), Normalize(p=1), HorizontalFlip(p =0.6), 
                               VerticalFlip(p=0.6), RandomBrightnessContrast(p=0.4), GaussNoise(p= 0.3),OpticalDistortion(p=0.3),
                               MotionBlur(p=0.3), HueSaturationValue(p=0.3)], p = 1)

    training_dataset = robotic_data_segmentation(training_images, transform=train_transform, problem_type=problem_type)
    first_item = training_dataset[0]

    features, labels = first_item 
    print(features.shape)   

    training_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory= torch.cuda.is_available())


    validation_dataset = robotic_data_segmentation(validation_images, transform=validation_transform,  problem_type=problem_type)
  

    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory= torch.cuda.is_available()) 
    print(training_dataset)
    print(len(training_data_loader))
    
    optimizer = Adam(model.parameters(), lr = learning_rate)

    model_directory = model_infor_directory_path/f'model_{str(fold)}.pt'

    try:
        state = torch.load(str(model_directory))
        epoch = state["epoch"]
        step = state['step']
        #model.load_from_state_dict(state["model"])
        model.load_state_dict(state["model"])

    except FileNotFoundError:
        epoch, step = 0, 0 

  

    log_file_name = f"train_{str(fold)}.log"

    log_file_name = model_infor_directory_path.joinpath(log_file_name)

    with log_file_name.open("a", encoding = 'utf8') as progress_notes:
        validation_loss_list = []
        for epoch in range(epoch,total_epochs):
            model.train()
            random.seed()
            total_size = len(training_data_loader)*batch_size
            bar = progressbar.ProgressBar(maxval= total_size)
            bar.start()
            loss_list = []
            
            averageloss = 0
            for i, (image, mask) in enumerate(training_data_loader):
                image = image.cuda() if torch.cuda.is_available() else image
                optimizer.zero_grad()

                with torch.no_grad():
                    mask = mask.cuda() if torch.cuda.is_available() else mask

                predictions = model(image)

                if problem_type == 'binary':
                    loss = binary_cross_entropy_with_iou(predictions, mask)

                elif  problem_type == 'parts':
                    loss = multiclass_cross_entropy_with_iou(predictions, mask, num_classes)
                else:
                    loss = multiclass_cross_entropy_with_iou(predictions, mask, num_classes)
                loss.backward()

                optimizer.step()

                step+=1


                loss_list.append(loss.item())

                bar.update((i + 1) * batch_size)

                print(loss)

                if i % 20 == 0:
                    averageloss = np.mean(loss_list)
                    information = {'loss': averageloss}  
                    loss_list = []
                    information['step'] = step
                    current_datetime = datetime.now()
                    datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    information['dt'] = datetime_string
                    progress_notes.write(json.dumps(information, sort_keys=True))
                    progress_notes.write('\n')
                    progress_notes.flush()
                    
            bar.finish()

            torch.save({'model': model.state_dict(), "epoch": epoch, "step" : step}, model_directory)

            if problem_type == 'binary':
                val_scores = val_bin(model, binary_cross_entropy_with_iou, validation_data_loader, num_classes)
            else:
                val_scores = val_multi(model, multiclass_cross_entropy_with_iou, validation_data_loader, num_classes)

    
            val_scores['step']= step
            progress_notes.write(json.dumps(val_scores, sort_keys=True))
            progress_notes.write('\n')
            progress_notes.flush()
            valid_loss = val_scores['valid_loss']
            validation_loss_list.append(valid_loss)          


if __name__ == '__main__':
    main()