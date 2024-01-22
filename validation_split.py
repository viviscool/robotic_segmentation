
def four_fold_cross_val(factor):
    val_dict  = {0: [1,8], 1: [5,6], 2: [3,7], 3: [2,4]}

    data_location = data_path / 'cropped_images'

    for x in range(1,9):

        validation_images =[]


        training_images = []

        if x in val_dict[factor]:
            validation_images.extend(list((data_location / ('instrument_dataset_' + str(x))/'training_images').glob('*')))

        else:
            training_images.extend(list((data_location / ('instrument_dataset_' + str(x))/'training_images').glob('*')))

    return training_images, validation_images

