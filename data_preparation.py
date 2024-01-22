# from pathlib import Path
# from tqdm import tqdm
# from rich.progress import Progress
# import cv2
# import numpy as np 

# #path to file with project data 

# current_working_directory = Path.cwd()

# data_directory = current_working_directory/'data'


# #path to file with training data 
# training_set = data_directory/'training_set'

# cropped_images = data_directory/'cropped_images'


# image_width, image_height = 1920, 1080

# x_start, y_start, new_width, new_height = 320, 28, 1280, 1024

# binary_classification = 255

# instrument_part_class = 85

# instrument_type = 32

# for x in range(1,9):
#     folder_number = 'instrument_dataset_' + str(x)
    
#     ((cropped_images/folder_number/'training_image_data').mkdir(parents=True, exist_ok=True))

#     (cropped_images/folder_number/'binary_masks').mkdir(parents=True, exist_ok=True)

#     (cropped_images/folder_number/'parts_masks').mkdir(parents=True, exist_ok=True)

#     (cropped_images/folder_number/'instruments_masks').mkdir(parents=True, exist_ok=True)
     
#     #list of instrument type from label folders 
#     label_folders = list((training_set/folder_number/'ground_truth').glob('*'))

 

#     with Progress() as progress:

#         task = progress.add_task("[green]Pre-processing images...", total=len(list((training_set / folder_number / 'left_frames').glob('*'))))

#         for x in list((training_set / folder_number / 'left_frames').glob('*')):
#             image = cv2.imread(str(x))

#             print(image.shape)
            
#             x_image = image[y_start: y_start + new_height, x_start: x_start + new_width]

#             cv2.imwrite(str(cropped_images/folder_number/'training_image_data'/(x.stem + '.jpg')), x_image)
            
#             truth_binary = np.zeros((image_height, image_width))

#             truth_parts = np.zeros((image_height, image_width))

#             truth_instruments = np.zeros((image_height, image_width))

#             for y in label_folders:
#                 mask = cv2.imread(str(y/x.name), 0)

#                 if 'Bipolar_Forceps' in str(y):
#                     truth_instruments[mask > 0] = 1
#                 elif 'Prograsp_Forceps' in str(y):
#                     truth_instruments[mask > 0] = 2
#                 elif 'Large_Needle_Driver' in str(y):
#                     truth_instruments[mask > 0] = 3
#                 elif 'Vessel_Sealer' in str(y):
#                     truth_instruments[mask > 0] = 4
#                 elif 'Grasping_Retractor' in str(y):
#                     truth_instruments[mask > 0] = 5
#                 elif 'Monopolar_Curved_Scissors' in str(y):
#                     truth_instruments[mask > 0] = 6
#                 elif 'Other' in str(y):
#                     truth_instruments[mask > 0] = 7
                
#                 if 'Other' not in str(y):
#                     truth_binary+=mask

#                     truth_parts[mask == 10] = 1
#                     truth_parts[mask == 20] = 2
#                     truth_parts[mask == 30] = 3
            
#             truth_binary[y_start: y_start + new_height, x_start: x_start + new_width] = (truth_binary[y_start: y_start + new_height, x_start: x_start + new_width] > 0).astype(np.uint8) * 255

#             truth_parts = (truth_parts[y_start: y_start + new_height, x_start: x_start + new_width]).astype(np.uint8)*instrument_part_class
#             truth_instruments = (truth_instruments[y_start: y_start + new_height, x_start: x_start + new_width]).astype(np.uint8)*instrument_type   

#             cv2.imwrite(str(cropped_images/folder_number/'binary_masks'/(x.stem + '.jpg')), truth_binary)
#             cv2.imwrite(str(cropped_images/folder_number/'parts_masks'/(x.stem + '.jpg')), truth_parts)
#             cv2.imwrite(str(cropped_images/folder_number/'instruments_masks'/(x.stem + '.jpg')), truth_instruments)


from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def create_directories(base_dir, subfolders):
    """Create directories for storing processed data."""
    for folder in subfolders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)

def process_image(file_path, base_dir, label_folders, coords):
    """Process an individual image file."""
    x_start, y_start, new_width, new_height = coords
    image = cv2.imread(str(file_path))
    cropped_image = image[y_start: y_start + new_height, x_start: x_start + new_width]

    cv2.imwrite(str(base_dir / 'training_image_data' / (file_path.stem + '.png')), cropped_image)

    # Initialize truth arrays
    truth_binary = np.zeros((new_height, new_width), dtype=np.uint8)
    truth_parts = np.zeros((new_height, new_width), dtype=np.uint8)
    truth_instruments = np.zeros((new_height, new_width), dtype=np.uint8)

    # Process each label folder
    for label_folder in label_folders:
        mask = cv2.imread(str(label_folder / file_path.name), 0)

        
        cropped_mask = mask[y_start: y_start + new_height, x_start: x_start + new_width]
        
        # Update truth arrays based on conditions
        # These conditions will need to be adapted to your specific case
        if 'Bipolar_Forceps' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 1
        elif 'Prograsp_Forceps' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 2
        elif 'Large_Needle_Driver' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 3
        elif 'Vessel_Sealer' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 4
        elif 'Grasping_Retractor' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 5
        elif 'Monopolar_Curved_Scissors' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 6
        elif 'Other' in str(label_folder):
            truth_instruments[cropped_mask > 0] = 7
        
        if 'Other' not in str(label_folder):
            truth_binary += cropped_mask

            truth_parts[cropped_mask == 40] = 0
            truth_parts[cropped_mask == 10] = 1
            truth_parts[cropped_mask == 20] = 2
            truth_parts[cropped_mask == 30] = 3

    truth_binary[truth_binary>0] = 1
    
    truth_binary = (truth_binary.astype(np.uint8)) * 255
    truth_parts = (truth_parts.astype(np.uint8))*85
    truth_instruments = (truth_instruments.astype(np.uint8))*32

    # Save the masks
    cv2.imwrite(str(base_dir / 'binary_masks' / (file_path.stem + '.png')), truth_binary)
    cv2.imwrite(str(base_dir / 'parts_masks' / (file_path.stem + '.png')), truth_parts)
    cv2.imwrite(str(base_dir / 'instruments_masks' / (file_path.stem + '.png')), truth_instruments)

def main():
    """Main function to set up and execute parallel image processing."""
    current_working_directory = Path.cwd()
    data_directory = current_working_directory/'data'
    training_set = data_directory / 'training_set'
    cropped_images = data_directory / 'cropped_images'

    coords = (320, 28, 1280, 1024)
    subfolders = ['training_image_data', 'binary_masks', 'parts_masks', 'instruments_masks']

    for x in range(1, 9):
        folder_number = f'instrument_dataset_{x}'
        folder_path = cropped_images / folder_number
        create_directories(folder_path, subfolders)

        label_folders = list((training_set / folder_number / 'ground_truth').glob('*'))

        with ThreadPoolExecutor() as executor:
            for file_path in (data_directory/ 'training_set' / folder_number / 'left_frames').glob('*'):
                executor.submit(process_image, file_path, folder_path, label_folders, coords)

if __name__ == "__main__":
    main()
