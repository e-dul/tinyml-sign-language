import argparse
import serial
import cv2 as cv
import os.path as path
import numpy as np

START_TOKEN = b'\x00\x01\x00\x01'
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def readserial(comport, baudrate, img_w, img_h, preview_scale_factor, save_dir):
    expected_len = img_w * img_h
    serial_comm = serial.Serial(comport, baudrate, timeout=0.1)
    idx = 0
    img_data_buffer = []
    
    while True:
        data = serial_comm.readall()

        if data == START_TOKEN:
            print("START")
            img_data_buffer = []
        else:
            img_data_buffer.extend([d for d in data]) 
        
        if len(img_data_buffer) >= expected_len:
            image = np.array(img_data_buffer[:expected_len].copy(), dtype="uint8") # Grayscale
            label_id = img_data_buffer[expected_len]
            pred = img_data_buffer[expected_len + 1]
            print(f"{label_id}  {pred} %")
            img_data_buffer = img_data_buffer[expected_len + 2:]
            # text = f"{LABELS[label_id]}  {pred} %"
            text = f"{label_id}  {pred} %"
            image = np.reshape(image,(img_h, img_w,1))
            if save_dir and path.exists(save_dir):
                cv.imwrite(path.join(save_dir, f"test{idx}.jpg"), image)
                idx = idx + 1
            
            image = cv.resize(image, None, fx=preview_scale_factor, fy=preview_scale_factor)
            image = cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 255, 255) , 2, cv.LINE_AA) 
            cv.imshow("live", image)
            cv.waitKey(100)
            

        if data:
            print(len(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='live stream tool - preview and capture images from serial')
    parser.add_argument(
        '--port', type=str, default="COM6", help='Serial port')
    parser.add_argument(
        '--baudrate', type=int, default=115200, help='Baudrate')
    parser.add_argument(
        '--img_w', type=int, default=176, help='Expected image width')
    parser.add_argument(
        '--img_h', type=int, default=144, help='Expected image height')
    parser.add_argument(
        '--preview_scale_factor', type=float, default=4.0, help='Requested preview scale factor comparing to original image size')
    parser.add_argument(
        '--save_dir', type=str, default="", help='Directory to save data. If not empty each image will be saved.')

    args = parser.parse_args()
    readserial(args.port, args.baudrate, args.img_w, args.img_h, args.preview_scale_factor, args.save_dir)
