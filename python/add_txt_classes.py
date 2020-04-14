import os

IMAGES_FOLDER = "labelled-images"
IMAGE_FORMATS = ["jpg", "jpeg", "png"]

files = list(os.walk(IMAGES_FOLDER))[0][-1]
images = [file for file in files if file.split(".")[-1] in IMAGE_FORMATS]
texts = [file for file in files if file.split(".")[-1] not in IMAGE_FORMATS]

for image in images:
    text_file = image.split(".")[0]+".txt"
    if text_file not in texts:
        with open(os.path.join(IMAGES_FOLDER, text_file), "w") as file:
            file.write("")

with open(os.path.join(IMAGES_FOLDER, "classes.txt"), "r") as file:
    classes = file.read()

with open("classes.names", "w") as file:
    file.write(classes)