# taken from https://medium.com/@a.ayyuced/image-classification-models-on-arduino-nano-33-ble-sense-60bf845fd2aa
import argparse
import binascii
import os
from pyexpat import model


def convert_to_c_array(bytes) -> str:
  hexstr = binascii.hexlify(bytes).decode("UTF-8")
  hexstr = hexstr.upper()
  array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
  array = [array[i:i+10] for i in range(0, len(array), 10)]
  return ",\n  ".join([", ".join(e) for e in array])


parser = argparse.ArgumentParser(description='Convert TFLite model to header')
parser.add_argument('model')
parser.add_argument('--output', default="model_out.h")

args = parser.parse_args()

if os.path.exists(args.model):
    tflite_binary = open(args.model, 'rb').read()
    ascii_bytes = convert_to_c_array(tflite_binary)
    c_file = "const unsigned char tf_model[] = {\n  " + ascii_bytes + "\n};\nunsigned int tf_model_len = " + str(len(tflite_binary)) + ";"
    open(args.output, "w").write(c_file)
else:
    print(f"{args.model} doesn't exist")