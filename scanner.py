import time
import cv2
import threading
import asyncio

class Scanner:
  def __init__(self, end, frame):
    self.end = end
    self.frame = frame.copy()  # Hacer una copia del frame

  def openScanner(self):
    start = 0  # Inicio de los segundos
    while start != self.end:
      time.sleep(1)
      start += 1
      cv2.putText(img=self.frame, text=f"{start}", org=(590, 160), fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=2, color=(0, 0, 0), thickness=2)
      cv2.imshow("frame", self.frame)
      cv2.waitKey(1)


