from ast import arg
from pathlib import Path
from  argparse import ArgumentParser
import cv2
import numpy as np

def multiclass_dice (cropped_ground_truth, cropped_prediction):

    if cropped_ground_truth.sum() == 0:
        return 1 if cropped_prediction.sum() == 0 else 0
    
    dice_scores_list = []

    for class_id in set(cropped_ground_truth.flatten()): 
        if class_id != 0: 
            ground_truth = (cropped_ground_truth == class_id)
            prediction = (cropped_prediction == class_id)
            dice_scores = [(2*(ground_truth*prediction).sum() + 1e-15 )/(ground_truth.sum() + prediction.sum() + 1e-15)]
            dice_scores_list.extend(dice_scores)
        else: 
            continue 

    return np.mean(dice_scores_list)
    

def multiclass_iou(cropped_ground_truth, cropped_prediction):
    if cropped_ground_truth.sum() == 0:
        return 1 if cropped_prediction.sum() == 0 else 0
     
    iou_scores_list = []

    for class_id in set(cropped_ground_truth.flatten()): 
        if class_id != 0: 
            ground_truth = (cropped_ground_truth == class_id)
            prediction = (cropped_prediction == class_id)
            intersection = (ground_truth*prediction).sum()
            union = ground_truth.sum() + prediction.sum() - intersection
            iou_scores = [(intersection + 1e-15)/(union+ 1e-15)]
            iou_scores_list.extend(iou_scores)
        else: 
            continue

    return np.mean(iou_scores_list) 



def main():
    parser = ArgumentParser(description="Evaluates segmentation model")
    parser.add_argument("--ground_truth_location", default= "data/test_ground_truth", help="Path of model")
    parser.add_argument("--prediction_location", default="data/predictions", help="Model type")
    parser.add_argument("--problem_type", default="binary")
    args = parser.parse_args()

    dice_list= []

    iou_list = []

    if args.problem_type == 'binary':
        for x in range(1,11):
            folder = "instrument_dataset_" + str(x)
            for x in Path(args.ground_truth_location/folder/"BinarySegmentation").glob("*"):
                ground_truth = (cv2.imread(str(x), 0) > 0).astype(np.uint8)

                cropped_ground_truth = ground_truth[28: 28 + 1024, 320: 320 + 1280]
                prediction_image = (cv2.imread(str(args.prediction_location/folder/"BinarySegmentation"/x.name), 0) > 127.5).astype(np.uint8)

                cropped_prediction = prediction_image[28: 28 + 1024, 320: 320 + 1280]

                dice = (2*(cropped_ground_truth*cropped_prediction).sum() + 1e-15 )/(cropped_ground_truth.sum() + cropped_prediction.sum() + 1e-15)

                dice_list.append(dice)

                intersection = (cropped_ground_truth*cropped_prediction).sum()

                union = cropped_ground_truth.sum() + cropped_prediction.sum() - intersection

                iou = (intersection + 1e-15)/(union+ 1e-15)

                iou_list.append([iou])

    elif args.problem_type == 'parts':
        for x in range(1,11):
            folder = "instrument_dataset_" + str(x)
            for x in Path(args.ground_truth_location/folder/"PartsSegmentation").glob("*"):
                ground_truth = cv2.imread(str(x), 0)

                cropped_ground_truth = ground_truth[28: 28 + 1024, 320: 320 + 1280]

                prediction_image = cv2.imread(str(args.prediction_location/folder/"PartsSegmentation"/x.name), 0)

                cropped_prediction = prediction_image[28: 28 + 1024, 320: 320 + 1280]

                dice_list.extend([multiclass_dice(cropped_ground_truth, cropped_prediction)])

                iou_list.extend([multiclass_iou(cropped_ground_truth, cropped_prediction)])

    else:
        for x in range(1,11):
            folder = "instrument_dataset_" + str(x)
            for x in Path(args.ground_truth_location/folder/"TypeSegmentation").glob("*"):
                ground_truth = cv2.imread(str(x), 0)

                cropped_ground_truth = ground_truth[28: 28 + 1024, 320: 320 + 1280]

                prediction_image = cv2.imread(str(args.prediction_location/folder/"TypeSegmentation"/x.name), 0)

                cropped_prediction = prediction_image[28: 28 + 1024, 320: 320 + 1280]

                dice_list.extend([multiclass_dice(cropped_ground_truth, cropped_prediction)])

                iou_list.extend([multiclass_iou(cropped_ground_truth, cropped_prediction)])


    print(f"Dice score for {str(args.problem_type)} segmentation is {str(np.mean(dice))}")

    print(f"IOU score for {str(args.problem_type)} segmentation is {str(np.mean(iou))}")
    


if __name__ == "__main__":
    main()