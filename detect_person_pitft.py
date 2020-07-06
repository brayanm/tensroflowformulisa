import time
import numpy as np
import cv2
import picamera
import picamera.array
import pygame
import os
from datetime import datetime
import RPi.GPIO as GPIO
from object_detection import ObjectDetection
import tensorflow as tf
import PIL.Image as Image


#set gpio for Adafruit Pitft 2.2 (4 buttons)
GPIO.setmode(GPIO.BCM)

GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)

old_27 = 1
old_23 = 1
old_22 = 1
old_17 = 1


x = 0
y = 0

#os environ for Adafruit Pitft 2.2
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)
os.environ['SDL_FBDEV'] = "/dev/fb0"
os.environ['SDL_VIDEODRIVER'] = "fbcon"


#path models
MODEL_FILENAME_person = '/home/pi/Documents/object_detection/model_person/includes/model_person.tflite'
LABELS_FILENAME_person = '/home/pi/Documents/object_detection/model_person/includes/labels_person.txt'
MODEL_LOCAL = "/home/pi/Documents/object_detection/model_person/includes/detect_person.tflite"


CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

#class from custom vision ObjectDetection
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


#function for draw text in display with pygame        
def blit_text2(surface, text, x, y, font, font_size, color):
    words = text.split('\n') # 2D array where each row is a list of words.
    max_width, max_height = surface.get_size()
    space = font.size(' ')[0]  # The width of a space.
    for line in words:
        word_surface2 = font.render(line, True, pygame.Color(color))
        word_width, word_height = word_surface2.get_size()
        word_surface3 = pygame.transform.rotate(word_surface2, 180)
        new_x = max_width - word_width
        surface.blit(word_surface3, (new_x, y))
        y = y - font_size



def run(opt, model):
    with picamera.PiCamera() as camera:
        # Set the camera resolution
        w = 1920
        h = 1080
        camera.resolution = (w, h)
        #camera.sensor_mode = 4
        camera.rotation = 90
        camera.framerate = 1
        #camera.vflip = True
        #camera.hflip = True
        # camera.awb_mode = 'off'
        # camera.awb_gains = (0.5, 0.5)

        # Need to sleep to give the camera time to get set up properly
        time.sleep(1)

        with picamera.array.PiRGBArray(camera) as stream:
            # Loop constantly
            while True:
                if not(GPIO.input(27)):
                    t_menu = "Seleccione Opción:\n1: Detector cv\n2: Detector formulisa\n4: Volver al Menu\n"
                    display.fill(black)
                    blit_text2(display, t_model, 50, 370, font2, 50, 'white')
                    pygame.display.update()
                    with open('/home/pi/Documents/contador', 'a') as file:
                        file.write(str(counter))
                    break
                # Grab data from the camera, in colour format
                # NOTE: This comes in BGR rather than RGB
                camera.capture(stream, format='bgr', use_video_port=True)
                image = stream.array

                # Display
                #resize, flip and rotate image
                newimg_save = cv2.resize(image,(int(640),int(480)))
                flipHorizontal = cv2.flip(newimg_save, 1)
                newimg_disp = cv2.resize(image,(display.get_width(), display.get_height()))
                surf = pygame.surfarray.make_surface(newimg_disp)
                surf = pygame.transform.flip(surf, True, False)
                surf = pygame.transform.rotate(surf, 270)
                w = display.get_width()
                h = display.get_height()
                display.blit(surf, (0, 0))
                if opt==1:
                    img2 = Image.fromarray(flipHorizontal)
                    #make prediction (custom vision)
                    prediction = model.predict_image(img2)
                    for p in prediction:
                        name = p['tagName']
                        conf = p['probability']
                        if conf > CONFIDENCE_THRESHOLD:
                            #draw boundingbox
                            x_min = p['boundingBox']['left']
                            y_min = p['boundingBox']['top']
                            x_max = p['boundingBox']['width']
                            y_max = p['boundingBox']['height']
                            y_min2 = int(max(1, (y_min * h)))
                            x_min2 = int(max(1, (x_min * w)))
                            y_max2 = int(min(h, (y_max * h)))
                            x_max2 = int(min(w, (x_max* w)))
                            pygame.draw.rect(display, (255, 0, 0), (x_min2, y_min2, x_max2, y_max2), 3)
                            
                            print("Detected", name)
                        blit_text2(display, name, 0, display.get_height()-30, font, 24, 'red')
                elif opt==2:
                    new_img = cv2.resize(flipHorizontal, (300, 300))
                    img2 = Image.fromarray(new_img)
                    #make prediction (formulisa)
                    model.set_tensor(input_details[0]['index'], [img2])

                    model.invoke()
                    rects = model.get_tensor(output_details[0]['index'])

                    scores = model.get_tensor(output_details[2]['index'])
                    print(scores)
                    
                    for index, score in enumerate(scores[0]):
                        if score > CONFIDENCE_THRESHOLD and score<1.0:
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
                            
                            print("Detected", "persona")
                        blit_text2(display, "persona", 0, display.get_height()-30, font, 24, 'red')
                pygame.display.update()

                stream.truncate(0)

                # If we press ESC then break out of the loop
                c = cv2.waitKey(7) % 0x100
                if c == 27:
                    break

    # Important cleanup here!
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # initialize the display
    pygame.init()
    display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    t_menu = "Seleccione Opción:\n1: Detector cv\n2: Detector formulisa\n4: Volver al Menu\n"

    font = pygame.font.SysFont('Arial', 24)
    font2 = pygame.font.SysFont('Arial', 50)
    black = (0,0,0)

    pygame.mouse.set_visible(False)
    
    display.fill(black)
    
    #model custom vision
    with open(LABELS_FILENAME_person, 'r') as f:
        labels_person = [l.strip() for l in f.readlines()]
    person_model = TFLiteObjectDetection(MODEL_FILENAME_person, labels_person)

    #model formulisa
    interpreter = tf.lite.Interpreter(model_path=MODEL_LOCAL)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    blit_text2(display, t_menu, 50, 370, font2, 50, 'white')

    pygame.display.update()

    # making a loop for buttons
    while True:  
        new_27 = GPIO.input(27)
        new_23 = GPIO.input(23)
        new_22 = GPIO.input(22)
        new_17 = GPIO.input(17)
        if not(new_17) and old_17 == 1:
            t_model = "Iniciando Detector cv,\npor favor espere!"
            display.fill(black)
            blit_text2(display, t_model, 0, 250, font2, 50, 'white')
            pygame.display.update()
            print('detector custom vision!')
            run(1, person_model)
            time.sleep(0.1)
        if not(new_22) and old_22 == 1:
            t_model = "Iniciando Detector formulisa,\npor favor espere!"
            display.fill(black)
            blit_text2(display, t_model, 0, 250, font2, 50, 'white')
            pygame.display.update()
            print('detector formulisa!')
            run(2, interpreter)
            time.sleep(0.1)
        if not(new_27) and old_27 == 1:
            t_model = "Seleccione Opción:\n1: Detector cv\n2: Detector formulisa\n4: Volver al Menu\n"
            display.fill(black)
            blit_text2(display, t_model, 50, 370, font2, 50, 'white')
            pygame.display.update()
            print('MENU!')
            time.sleep(0.1)
        old_27 = new_27
        old_23 = new_23
        old_22 = new_22
        old_17 = new_17
    GPIO.cleanup()
    
    
