import cv2
import uuid
import os


class CaptureImageData:
    def __init__(self, pos_path, anc_path):
        self._cap = cv2.VideoCapture(0)
        self._pos_path = pos_path
        self._anc_path = anc_path
    
    def run(self):
        while self._cap.isOpened():
            _, frame = self._cap.read()

            frame = frame[120:120+250, 200:200+250, :]
            
            # Colect the anchors imagss
            if cv2.waitKey(1) & 0XFF == ord('a'):
                imgname = os.path.join(self._pos_path, f'{uuid.uuid1()}.jpg')
                cv2.imwrite(imgname, frame)
            # Collect the positives images
            if cv2.waitKey(1) & 0XFF == ord('p'):
                imgname = os.path.join(self._anc_path, f'{uuid.uuid1()}.jpg')
                cv2.imwrite(imgname, frame)
            
            cv2.imshow('Image Collection', frame)

            if cv2.waitKey(1) & 0XFF == ord('q'):
                break

        self._cap.release()
        cv2.destroyAllWindows()