import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import pygame
import cv2
import pygame

#change for path of your model
MODEL_LOCAL = "/home/pi/Documents/object_detection/model_person/includes/detect_person.tflite"


def main(interpreter):
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
    new_img = cv2.resize(image, (300, 300))
    img2 = Image.fromarray(new_img)
    #make prediction (formulisa)
    model.set_tensor(input_details[0]['index'], [img2])

    model.invoke()
    rects = model.get_tensor(output_details[0]['index'])

    scores = model.get_tensor(output_details[2]['index'])
    print(scores)
    
    for index, score in enumerate(scores[0]):
        if score > 0.5 and score<1.0:
            #draw boundingbox
            x_min = rects[0][index][1]
            y_min = rects[0][index][0]
            x_max = rects[0][index][3]
            y_max = rects[0][index][2]
            y_min2 = int(max(1, (y_min * h)))
            x_min2 = int(max(1, (x_min * w)))
            y_max2 = int(min(h, (y_max * h)))
            x_max2 = int(min(w, (x_max* w)))
            pygame.draw.rect(display, (255, 0, 0), (x_min2, y_min2, x_max2, y_max2), 3)
    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()


if __name__ == '__main__':
    #model formulisa
    interpreter = tf.lite.Interpreter(model_path=MODEL_LOCAL)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    main(interpreter)