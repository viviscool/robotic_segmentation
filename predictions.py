from argparse import ArgumentParser
from ast import arg
from hmac import new
from re import U
from data_loader import robotic_data_segmentation
import cv2
from models import UNet16, UNet 
import torch 
import progressbar
import numpy as np  
import data_preparation
from torch.utils.data import DataLoader
from torch.nn import functional as F    
from albumentations import Compose, Normalize
from training_models import four_fold_cross_validation
import pathlib




def prepare_test_images(test_images):
     image_list =[]
     for x in range(1,11):
          image_list.extend(list((test_images / ('instrument_dataset_' + str(x))/'test_set').glob('*')))
     
     return image_list



def architecture(location, model = 'UNet', assessment = 'binary'):
     if assessment == "binary":
          classes = 1
     elif assessment == "instruments":
          classes = 8
     else:
          classes = 4
     
     if model == "UNet":
          model = UNet(num_classes = classes)
     
     else: 
          model = UNet16(num_classes = classes)   

     model_weights = torch.load(location)

     new_state ={}
     
     for key, value in model_weights.items():
          new_key = key.replace("module.", "")
          new_state[new_key] = value

     model.load_state_dict(new_state)
     
     if torch.cuda.is_available():
          return model.cuda()

     model.eval()

     return model 

def model_assessment(model, original_location, new_location, assessment = "binary"): 
     transform_data = Compose([Normalize(p=1)], p=1)
     data_set = robotic_data_segmentation(original_location, mode = 'make_predictions', transform = transform_data, problem_type = assessment)
     data = DataLoader(data_set, batch_size = 6, shuffle = False, num_workers = 12, pin_memory = torch.cuda.is_available())  

     with torch.inference_mode():
          bar = progressbar.ProgressBar(maxval= len(data))
          bar.start()
          for x, (image, file_names) in data:
               image = image.cuda() if torch.cuda.is_available() else image

               prediction = model(image)
               for y, file_name in enumerate(file_names):
                    if assessment == "binary":
                         multiplication = 255
                         selected_prediction = prediction[y, 0]
                         sigmoid_prediction = torch.sigmoid(selected_prediction)
                         numpy_prediction = sigmoid_prediction.detach().cpu().numpy()
                         result = numpy_prediction * multiplication.astype(np.uint8)
                         
                    elif assessment == "parts":
                         multiplication = 85
                         arg_max_output = torch.argmax(prediction[y], dim = 0)
                         scaled_output = arg_max_output * multiplication
                         numpy_prediction = scaled_output.cpu().detach()
                         result = numpy_prediction.numpy().astype(np.uint8)

                    elif assessment == "type":
                         multiplication = 32
                         prediction = torch.argmax(prediction[y], dim = 0)
                         scaled_output = prediction * multiplication
                         numpy_prediction = scaled_output.cpu().detach()
                         result = numpy_prediction.numpy().astype(np.uint8)

                    prediction_image = np.zeros(1080,1920)

                    prediction_image[28: 28 + 1024, 320: 320 + 1280] = result

                    current_path = pathlib.Path(file_name[y])

                    folder_number = current_path[1].name

                    new_directory_path = new_location/folder_number

                    new_directory_path.mkdir(parents=True, exist_ok=True)

                    output_file_path = new_directory_path/f"{current_path.stem}.png"

                    cv2.imwrite(str(output_file_path), prediction_image)
                    
                    bar.update(x+1)
          bar.finish()


def main():
     parser = ArgumentParser(description="Trains segmentation model")
     parser.add_argument("--model_location", default="model_info", help="Path of model")
     parser.add_argument("--model_type", default="UNet", help="Model type")
     parser.add_argument("--fold", default=0, help="Fold number")
     parser.add_argument("--assessment", default= "binary", help= "problem type")

     args = parser.parse_args()

     batch_size = 6

     workers = 12

     current_working_directory = pathlib.Path.cwd()

     test_images = current_working_directory/"data"/"cropped_test"

     prepared_test = prepare_test_images(test_images)

     save_images_path = current_working_directory/"predictions"/str(args.model_type)/"args.assessment"

     save_images_path.parent.mkdir(parents=True, exist_ok=True)

     location = str(args.model_location)

     if str(args.fold) == "all":
          for b in [0,1,2,3]:
               model = architecture((location + "/model_fold{fold}"), model = args.model_type, assessment = args.assessment)

               model_assessment(model, prepared_test, save_images_path, data_preparation.transform_data, assessment = args.assessment)
     else: 
          model = architecture((location + "/model_fold{fold}"), model = args.model_type, assessment = args.assessment)

          model_assessment(model, prepared_test, save_images_path, data_preparation.transform_data, assessment = args.assessment)
          
if __name__ == '__main__':
    main()


