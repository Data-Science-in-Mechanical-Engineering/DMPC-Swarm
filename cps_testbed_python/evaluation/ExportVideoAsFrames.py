import cv2
from pathlib import Path
partition = 0.42
vidcap = cv2.VideoCapture(f'{Path.home()}/Movies/Crash1.mov')
length_total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length_total)
success, image = vidcap.read()
count = 0
while success:
	if count % 1 == 0:
		cv2.imwrite(f"{Path.home()}/Movies/Test/frame{count:05d}.jpg", image)     # save frame as JPEG file
		success,image = vidcap.read()
		print('Read a new frame: ', success)
	count += 1
	if count > length_total * partition:
		break
