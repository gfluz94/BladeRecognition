import os
import numpy as np

IMG_FORMATS = ["png", "jpg", "jpeg"]

validation_size = 0.15

path = "./labelled-images"
files = [file for file in list(os.walk(path))[0][-1] if file.split(".")[-1] in IMG_FORMATS]

num_validation = int(len(files)*validation_size)
num_train = len(files)-num_validation

np.random.shuffle(files)
train_files = files[:-num_validation]
validation_files = files[-num_validation:]

text=""
for file in train_files:
    text += os.path.join(os.getcwd(), path, file)+"\n"
with open("train.txt", "w") as train_file:
    train_file.write(text.strip())

text=""
for file in validation_files:
    text += os.path.join(os.getcwd(), path, file)+"\n"
with open("validation.txt", "w") as validation_file:
    validation_file.write(text.strip())