import os

SOURCE_PATH = "./labelled-images"
FILE = "classes.txt"

os.system("scp " + os.path.join(SOURCE_PATH, FILE) + " ./") 
os.system("mv " + FILE + "  classes.names")