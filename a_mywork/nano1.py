import sys
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq

# use either of the formats below to specifiy address of display computer
# sender = imagezmq.ImageSender(connect_to='tcp://jeff-macbook:5555')
sender = imagezmq.ImageSender(connect_to='tcp://127.0.0.1:5555')

nano_name = 'nano1'  # send RPi hostname with each image
nano_cam = VideoStream().start()
time.sleep(2.0)  # allow camera sensor to warm up
jpeg_quality = 50  # 0 to 100, higher is better quality, 95 is cv2 default
while True:  # send images as stream until Ctrl-C
    image = nano_cam.read()
    ret_code, jpg_buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_jpg(nano_name, jpg_buffer)
