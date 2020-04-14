import os

CLASSES_FILE = "classes.names"
WEIGHTS_FOLDER = "backup"
TARGET_PATH = "data"

with open(CLASSES_FILE, "r") as file:
    number_classes = len(file.readlines())

text = f"classes = {number_classes}\n"
text += f"train = {os.path.join(TARGET_PATH, 'train.txt')}\n"
text += f"valid = {os.path.join(TARGET_PATH, 'validation.txt')}\n"
text += f"names = {os.path.join(TARGET_PATH, 'classes.names')}\n"
text += f"backup = {os.path.join('/mydrive/customYOLOv3', WEIGHTS_FOLDER)}\n"

with open("labelled_data.data", "w") as file:
    file.write(text.strip())