import os
import shutil
import random

def distribute_train_validation_split(validation_size=0.2):

    all_images = os.listdir("input_from_kaggle/train/")
    random.shuffle(all_images)

    all_dogs = list(filter(lambda image: "dog" in image, all_images))
    all_cats = list(filter(lambda image: "cat" in image, all_images))

    index_to_split = int(len(all_dogs) - len(all_dogs) * validation_size)
    training_dogs = all_dogs[:index_to_split]
    validation_dogs = all_dogs[index_to_split:]
    training_cats = all_cats[:index_to_split]
    validation_cats = all_cats[index_to_split:]

    shutil.rmtree("input_for_model")
    os.makedirs("input_for_model/train/dogs/", exist_ok=True)
    os.makedirs("input_for_model/train/cats/", exist_ok=True)
    os.makedirs("input_for_model/validation/dogs/", exist_ok=True)
    os.makedirs("input_for_model/validation/cats/", exist_ok=True)

    copy_images_to_dir(training_dogs, "./input_for_model/train/dogs")
    copy_images_to_dir(validation_dogs, "./input_for_model/validation/dogs")
    copy_images_to_dir(training_cats, "./input_for_model/train/cats")
    copy_images_to_dir(validation_cats, "./input_for_model/validation/cats")

def copy_images_to_dir(images_to_copy, destination):
    for image in images_to_copy:
        shutil.copyfile(f"./input_from_kaggle/train/{image}", f"{destination}/{image}")

distribute_train_validation_split(0.25)