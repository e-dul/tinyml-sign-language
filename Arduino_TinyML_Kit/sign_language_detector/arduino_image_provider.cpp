/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include "Arduino_OV767X_TinyMLx.h"

constexpr int kSrcWidth = 176;
constexpr int kSrcHeight = 144;
constexpr int kSrcMode = QCIF;
constexpr float y_factor = kSrcHeight / 96.0;
constexpr float x_factor = kSrcWidth / 96.0;
byte data[kSrcWidth * kSrcHeight];

// Get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, float* image_data, bool live) {

  static bool g_is_camera_initialized = false;
  static bool serial_is_initialized = false;

  // Initialize camera if necessary
  if (!g_is_camera_initialized) {
    if (!Camera.begin(kSrcMode, GRAYSCALE, 5, OV7675)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    Camera.autoGain();
    Camera.autoExposure();
    // 0 - 65535
    // Camera.setExposure(100);
    // Camera.setGain(128); // 0 - 255
    // void setSaturation(int saturation); // 0 - 255
    // void setHue(int hue); // -180 - 180
    // void setBrightness(int brightness); // 0 - 255
    // void setContrast(int contrast); // 0 - 127
    // Camera.setBrightness(127);
    // Camera.setContrast(64);

    g_is_camera_initialized = true;
  }

  // Read camera data
  Camera.readFrame(data);

  int index = 0;
  for (int y = 0; y < 96; y++) {
    int src_y = y_factor * y;
    for (int x = 0; x < 96; x++) {
      int src_x = x_factor * x;
      image_data[index++] = static_cast<float>(data[(src_y * kSrcWidth) + src_x]); // convert TF input image to float
    }
  }

  if (live) {
    Serial.write(data, kSrcWidth * kSrcHeight);
    delay(50);
  }

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE