from picamera2 import Picamera2
import argparse
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time


LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def run(model):
  # Setup TF Lite
  interpreter = tflite.Interpreter(model_path=model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # shape
  input_shape = input_details[0]['shape']

  #Setup camera with continous AF
  picam2 = Picamera2()
  picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))

  picam2.start()
  time.sleep(1)
  picam2.set_controls({"AfMode": 2 ,"AfTrigger": 0})
  time.sleep(1)

  cv2.namedWindow("Frame")

  while True:
    image = picam2.capture_array()

    proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    proc = cv2.resize(proc,(96, 96)).reshape((1,96,96,1))
    interpreter.set_tensor(input_details[0]['index'], proc.astype("float32"))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    label_idx = output_data.argmax()
    label = LABELS[label_idx]
    pred = output_data[0][label_idx]
    text = f"{label}  {pred} %"
    print(text)
    image = cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 0, 255) , 2, cv2.LINE_AA)

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key != ord('q'):
        pass
    else:
        cv2.destroyAllWindows()
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on live camera stream')
    parser.add_argument('model', type=str, help='TF Lite model path')
    args = parser.parse_args()
    run(args.model)
