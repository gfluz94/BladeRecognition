import os

CLASSES_FILE = "classes.names"
WEIGHTS_FOLDER = "backup"

with open(CLASSES_FILE, "r") as file:
    number_classes = len(file.readlines())

full_path = os.getcwd()

text = f"classes = {number_classes}\n"
text += f"train = {os.path.join(full_path, 'train.txt')}\n"
text += f"valid = {os.path.join(full_path, 'validation.txt')}\n"
text += f"names = {os.path.join(full_path, 'classes.names')}\n"
text += f"backup = {WEIGHTS_FOLDER}\n"

with open("labelled_data.data", "w") as file:
    file.write(text.strip())