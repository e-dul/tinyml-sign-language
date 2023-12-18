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

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <TinyMLShield.h>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 105 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
bool live = false;
byte start [4] = {0,1,0,1};
byte metadata [2] = {0, 0};
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  // wait for serial to get logs :) 
  while(!Serial);

  initializeShield();

  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<12> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddSub();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddDequantize();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

 // Print out detailed allocation information:
 // interpreter->GetMicroAllocator().PrintAllocations();

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  // Serial.println(input->type);
}

// The name of this function is important for Arduino compatibility.
void loop() {

  if (readShieldButton()) {
    live = !live; 
    Serial.write(start, 4);
    delay(50);
  }

  int getMicros = millis();
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.f, live)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
    return;
  }
  unsigned startMicros = millis();
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    return;
  }
  int endMicros = millis();
  // TF_LITE_REPORT_ERROR(error_reporter, "get image: %dms inference: %dms",
  //                      startMicros - getMicros, endMicros - startMicros);
  TfLiteTensor* output = interpreter->output(0);

  // Find the current highest scoring category.
  int current_top_index = kCategoryNothing;
  float current_top_score = 0.5; // min threshodl
  // float current_min_score = 100.0; 
  for (int i = 0; i < kCategoryCount; ++i) {
    float val = output->data.f[i];
    if (val > current_top_score) {
      current_top_score = val;
      current_top_index = i;
    }
  }
  if (current_top_index != kCategoryNothing) {
    const char* current_top_label = kCategoryLabels[current_top_index];
    // Serial.println(current_top_label);
    // Serial.println(current_top_score);
    // Serial.println(current_min_score);
    if (!live) {
      TF_LITE_REPORT_ERROR(error_reporter, "top 1: %d %s",
                          static_cast<int>(current_top_score * 100), current_top_label);
    }    
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }
  if (live) {
    metadata[0] = static_cast<uint8>(current_top_index);
    metadata[1] = static_cast<uint8>(current_top_score * 100);
    Serial.write(metadata, 2);
    delay(10);
  }
}
