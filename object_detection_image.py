import cv2
from PIL import ImageFont, ImageDraw, Image 
import numpy as np
import time
import os

IMAGE_PATH = "./test"
OUTPUT_PATH = "./test"
MODEL_PATH = "./backup"

def read_image(img_path):
    return cv2.imread(img_path)

def show_image(img, window_name="Object Detection"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def make_blob(img, to_shape=(416, 416), rgb=True):
    return cv2.dnn.blobFromImage(img, 1/255.0, to_shape, swapRB=rgb, crop=False)

class Yolo():

    def __init__(self, config_path, weights_path, labels_path):
        self._config_path = config_path
        self._weights_path = weights_path
        self._labels_path = labels_path

        self.model = self.build_model()
        self.labels = self.set_labels()

    def set_labels(self):
        with open(self._labels_path, "r") as file:
            labels = [label.strip() for label in file.readlines()]
        return labels

    def build_model(self):
        return cv2.dnn.readNetFromDarknet(self._config_path, self._weights_path)

    def output_layers(self):
        layers_names = self.model.getLayerNames()
        return [layers_names[i[0]-1] for i in self.model.getUnconnectedOutLayers()]

    def predict(self, blob):
        self.model.setInput(blob)
        return self.model.forward(self.output_layers())

def get_bounding_boxes(network_output, labels, img_shape, probability_threshold=0.5):
    bounding_boxes = []
    confidences = []
    class_numbers = []
    height, width = img_shape
    for results in network_output:
        for detection in results:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current>probability_threshold:
                x, y, w, h = detection[:4]*np.array([width, height, width, height])
                x_min, y_min = int(x-w/2), int(y-h/2)
                x_max, y_max = int(x+w/2), int(y+h/2)
                bounding_boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(confidence_current))
                class_numbers.append(int(class_current))
    return bounding_boxes, confidences, class_numbers

def non_max_suppresion(boxes, confidences, probability_threshold=0.5, nms_threshold=0.3):
    return cv2.dnn.NMSBoxes(boxes, confidences, probability_threshold, nms_threshold)

def show_detected_objects(img, nms_results, boxes, scores, classes, labels, save_img=True, file_name=None):
    counter = 1
    colors = [[252, 15, 192]]
    if nms_results:
        for i in nms_results.flatten():
            text_label = labels[classes[i]]
            print(f"Object {i}: {text_label}")
            counter+=1
            top_left = tuple(boxes[i][:2])
            bottom_right = tuple(boxes[i][2:])
            color = colors[classes[i]]
            cv2.rectangle(img, top_left, bottom_right, color, 2)
            show_text = f"{text_label}"#: {scores[i]:.3f}"
            pil_im = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_im) 
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=int(img.shape[1]/763*20))
            label_size = draw.textsize(show_text, font)
            draw.rectangle([(top_left[0], top_left[1]), (top_left[0]+label_size[0], top_left[1]-label_size[1])], fill="#FC0FC0")
            draw.text((top_left[0], top_left[1]-label_size[1]), show_text, font=font, fill=(255, 255, 255, 255))
            img = np.array(pil_im)
        if save_img:
            cv2.imwrite(os.path.join(OUTPUT_PATH, file_name), img)
    show_image(img, window_name="Object Detection")
    

if __name__=="__main__":
    # Reading Original Image
    img_name = "chandler.jpg"
    img = read_image(os.path.join(IMAGE_PATH, img_name))
    height, width = img.shape[:2]
    show_image(img, window_name="Original Image")

    # Getting blob out of image
    blob = make_blob(img, to_shape=(416, 416), rgb=True)

    # Getting Model and Predicting from blob
    labels_file = "./classes.names"
    config_file = os.path.join(MODEL_PATH, "yolov3_test.cfg")
    weights_file = os.path.join(MODEL_PATH, "yolov3_train_final.weights")

    yolo = Yolo(config_file, weights_file, labels_file)
    network_output = yolo.predict(blob)
    
    # Getting boxes and applying Non Maximum Suppression
    probability_threshold = 0.1
    boxes, scores, classes = get_bounding_boxes(network_output, yolo.labels, (height, width), probability_threshold)
    results = non_max_suppresion(boxes, scores, probability_threshold, nms_threshold=0.1)

    # Showing final result
    final_name = img_name.split(".")[0]+"-predicted."+img_name.split(".")[-1]
    show_detected_objects(img, results, boxes, scores, classes, yolo.labels, save_img=True, file_name=final_name)


    

