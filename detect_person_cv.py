# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import pygame
import cv2
import pygame

#change for path of your model
MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def main():
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

    pygame.init()
    screen = pygame.display.set_mode((640,480))
    #change for path of your image
    IMAGE_PATH = "/home/osboxes/Documents/tensorflow/workspace/training_demo/images/train/84.jpg"
    image = Image.open(IMAGE_PATH)
    I = np.asarray(Image.open(IMAGE_PATH))
    h = I.shape[0]
    w = I.shape[1]
    predictions = od_model.predict_image(image)
    print(predictions)
    surf = pygame.surfarray.make_surface(I)
    surf2 = pygame.transform.rotate(surf, 270)
    surf3 = pygame.transform.flip(surf2, True, False)
    screen.blit(surf3, (0, 0))
    if p['probability']>0.5:
        for p in predictions:
            x_min = p['boundingBox']['left']
            y_min = p['boundingBox']['top']
            x_max = p['boundingBox']['width']
            y_max = p['boundingBox']['height']
            y_min2 = int(max(1, (y_min * h)))
            x_min2 = int(max(1, (x_min * w)))
            y_max2 = int(min(h, (y_max * h)))
            x_max2 = int(min(w, (x_max* w)))
            pygame.draw.rect(screen, (255, 255, 255), (x_min2, y_min2, x_max2, y_max2), 3)
    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()


if __name__ == '__main__':
    main()