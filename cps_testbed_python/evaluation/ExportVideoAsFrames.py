import cv2
from pathlib import Path
vidcap = cv2.VideoCapture(f'{Path.home()}/Videos/Test1234.mp4')
success, image = vidcap.read()
count = 0
while success:
	if count % 1 == 0:
	  cv2.imwrite(f"{Path.home()}/Videos/Test/frame{count:05d}.jpg", image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  print('Read a new frame: ', success)
	count += 1