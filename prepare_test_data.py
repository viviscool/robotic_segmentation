from pathlib import Path

import cv2
import numpy as np

current_working_directory = Path.cwd()

test_data_location = (current_working_directory/'data'/"test_set")

cropped_test = (current_working_directory/"data"/ "cropped_test")


if __name__ == "__main__":
    for x in range (1,11):
        folder = "instrument_dataset_" + str(x)

        (cropped_test/folder/"test_image_data").mkdir(exist_ok =True, parents =True)

        for y in list((test_data_location/folder/"left_frames").glob("*")):
            test_image = cv2.imread(str(y))

            test_image = test_image[28: 28 + 1024, 320: 320 + 1280]
            
            cv2.imwrite(str(cropped_test / folder / "test_image_data" / (y.stem + '.png')), test_image)

