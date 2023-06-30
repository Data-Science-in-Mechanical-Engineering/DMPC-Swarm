import cv2
import os

if __name__=="__main__":
    folder = 'F:/Dokumente/Daten BA Anti Collision/Tests/test_video_pot'
    video_name = folder + '/APF_4et_gnron_30_drones_30fps.avi'
    image_folder = folder + '/images_v2'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Microspoft Video 1

    len_video = 2400

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith("ImgPath")]
    frame = cv2.imread(os.path.join(image_folder, "ImgPath_0.jpg"))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, fourcc, 30, (width,height), )

    for i in range(1, len_video, 2):
        video.write(cv2.imread(os.path.join(image_folder, "ImgPath_" + str(i) + ".jpg")))

    video.release()
