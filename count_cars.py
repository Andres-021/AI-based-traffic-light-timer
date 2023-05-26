import cv2
import torch 
import numpy as np
import matplotlib.path as mplPath
import time
from scanner import Scanner
import asyncio
import threading
from queue import Queue

ZONE = np.array([
  [33, 346],
  [438, 273],
  [452, 246],
  [454, 197],
  [435, 145],
  [419, 104],
  [146, 97],
  [96, 196],
  [28, 245],
  [34, 349],
])



def get_center_point(bbox):
  #xmin, ymin, xmax, ymax
  # 0      1     2     3
  center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
  return center


def load_model():
  model = torch.hub.load("ultralytics/yolov5", model="yolov5n", pretrained=True)
  return model


def get_bboxes(preds: object):
  #xmin, ymin, xmax, ymax
  df = preds.pandas().xyxy[0] # Tomamos el objeto preds tranformandolo a su formato pandas y obteniendolo en el formato boundi boxes xmin, ymin etc
  df = df[df["confidence"] >= 0.2] # Filtramos aquellos que tengan una probabilidad mayor a 0.5
  # df = df[df["name"] == "car"]
  # Filtramos los carros y las personas
  df = df.loc[(df["name"] == "car") | (df["name"] == "person")]

  return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int) # Solo retornamos los boundi boxes(cajas de vista), solo los valores


def is_valid_detection(xc, yc):
  return mplPath.Path(ZONE).contains_point((xc, yc))


def calculate_seconds(frame, seconds, detections: []):
  # Calcula carros, pasa el promedio de carros que seran los segundos y luego muestra el color
  # # 10 20 o 30 segundos
  # prom = sum(detections) // detections.length
  # # LUCES DEL SEMAFORO
  cv2.putText(img=frame, text=f"{seconds}", org=(590, 160), fontFace=cv2.FONT_HERSHEY_PLAIN,
    fontScale=2, color=(0, 0, 0), thickness=2)


  if len(detections) <= 2 :
    # VERDE 
    cv2.circle(img=frame, center=(590, 120), radius=20, color=(18, 185, 31), thickness=-1)
  else: 
    # ROJO 
    cv2.circle(img=frame, center=(590, 40), radius=20, color=(230, 24, 24), thickness=-1)

  # NARANJA 
  # cv2.circle(img=frame, center=(590, 80), radius=20, color=(232, 144, 25), thickness=-1)




def detector():
  cap = cv2.VideoCapture("data/cars.mp4")
  frame_rate = cap.get(cv2.CAP_PROP_FPS)

  decrement = 1
  factor_escala = int(frame_rate / decrement)

  total_seconds = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate)
  frame_counter = 0

  model  = load_model()

  prom = 0
  tem_prom = 0

  # SEMAFOROS
  red = False
  orange = False
  green = True

  while cap.isOpened():
    # Frame quiere decir los cuadros por segundo cuando se ejecuta el video
    status, frame = cap.read()
    if not status:
      break

    preds = model(frame)
    bboxes = get_bboxes(preds)

    detections = [] # Guarda vehiculo detectado en cada posicion
    count_detections = 0

    for box in bboxes:
      # Obtenemos el punto central de la caja que marca los carros y motos
      xc, yc = get_center_point(box)

      if(is_valid_detection(xc, yc)):
        # Guardamos las detecciones para sacar el promedio
        count_detections += 1
        detections.append(count_detections) # Incrementamos la cantidad de objetos detectados, ya sea carros o motos
      # Pintamos el punto
      cv2.circle(img=frame, center=(xc, yc), radius=3, color=(0, 255, 0), thickness=-1)
      # pintamos al frame los rectangulos en base a datos de cada bboxes, osea los vectores obtenidos
      cv2.rectangle(img=frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(255, 0, 0), thickness=1)

    # CREANDO EL DISEÑO DEL SEMAFORO
    cv2.rectangle(img=frame, pt1=(565, 15), pt2=(615, 145), color=(60, 60, 60), thickness=-1)
    cv2.circle(img=frame, center=(590, 40), radius=20, color=(0, 0, 0), thickness=-1)
    cv2.circle(img=frame, center=(590, 80), radius=20, color=(0, 0, 0), thickness=-1)
    cv2.circle(img=frame, center=(590, 120), radius=20, color=(0, 0, 0), thickness=-1)

    # MOSTRAMOS EL COLOR DE CADA SEMAFORO ACTIVO
    if red:
      # ROJO 
      cv2.circle(img=frame, center=(590, 40), radius=20, color=(0, 0, 255), thickness=-1)

    if orange:
      # NARANJA 
      cv2.circle(img=frame, center=(590, 80), radius=20, color=(251, 192, 40), thickness=-1)

    if green:
      # VERDE 
      cv2.circle(img=frame, center=(590, 120), radius=20, color=(50,205,50), thickness=-1)


    # MODIFICAMOS EL SEMAFORO DEL SIGUIENTE COLOR ACTIVO SI LOS SEGUNDOS PASARON A 0
    if prom == 0:
      # Calculamos los segundos segun la velocidad del video
      if frame_counter % factor_escala == 0:
        # Mostrar segundos
        prom = int(sum(detections) / len(detections))
        prom = int(prom * len(detections))

        if red:
          red = False
          orange = True
          green = False

        elif orange:
          red = False
          orange = False
          green = True

        elif green:
          red = True
          orange = False
          green = False


    cv2.putText(img=frame, text=f"{prom}", org=(580, 170), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 0), thickness=2)
    cv2.putText(img=frame, text=f"Vehiculos: {count_detections}", org=(400, 340), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 0), thickness=3)

    frame_counter += 1
    # Calculamos los segundos segun la velocidad del video
    if frame_counter % factor_escala == 0:
      prom -= 1

      # Decrementamos los segundos
      total_seconds -= 1
    
    # Mostramos la zona definida arriba
    # Poligonos para mostrar el cuadrado
    cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(0, 0, 255), thickness=1)
    # Mostramos
    cv2.imshow("frame", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


# if __name__ == '__main__':
#   cap = cv2.VideoCapture("data/cars.mp4")
#   frame_rate = cap.get(cv2.CAP_PROP_FPS)
detector()