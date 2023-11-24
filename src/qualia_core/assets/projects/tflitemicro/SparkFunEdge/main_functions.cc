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

extern "C" {
#include "am_bsp.h"  // NOLINT
}

#include "main_functions.h"

#include <time.h>

#include "output_handler.h"
#include "input_handler.h"
#include "model_data.h"
//#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 192 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int input_vector_length = 1;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);
  am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
  am_hal_cachectrl_enable();
  am_bsp_low_power_init();

  myapp_uart_init();
  myapp_printf("%s", "READY\r\n");
  //enable_burst_mode();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  //static tflite::MicroErrorReporter micro_error_reporter;
  static AMMicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    //error_reporter->Report(
    myapp_printf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
//  static tflite::ops::micro::AllOpsResolver resolver;

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
#if 0 // Old TF 2.2
  static tflite::MicroMutableOpResolver resolver;  // NOLINT
  //resolver.AddBuiltin(
  //    tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  //    tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  resolver.AddBuiltin(
      tflite::BuiltinOperator_ADD,
      tflite::ops::micro::Register_ADD(), 1, 2); // float32=v1, uint8=v2
  resolver.AddBuiltin(
      tflite::BuiltinOperator_MAX_POOL_2D,
      tflite::ops::micro::Register_MAX_POOL_2D(), 1, 2); // float32=v1, uint8=v2
  resolver.AddBuiltin(
      tflite::BuiltinOperator_CONV_2D,
      tflite::ops::micro::Register_CONV_2D(), 1, 3); // float32=v1, uint8=v3
  resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED(), 1, 4); // float32=v1, uint8=v4
  resolver.AddBuiltin(
      tflite::BuiltinOperator_QUANTIZE,
      tflite::ops::micro::Register_QUANTIZE(), 1, 1);
  resolver.AddBuiltin(
      tflite::BuiltinOperator_DEQUANTIZE,
      tflite::ops::micro::Register_DEQUANTIZE(), 2, 2);
  resolver.AddBuiltin(
      tflite::BuiltinOperator_PAD,
      tflite::ops::micro::Register_PAD(), 1, 2); // float32=v1, uint8=v2
  resolver.AddBuiltin(
      tflite::BuiltinOperator_RESHAPE,
      tflite::ops::micro::Register_RESHAPE(), 1, 1);
  resolver.AddBuiltin(
      tflite::BuiltinOperator_SOFTMAX,
      tflite::ops::micro::Register_SOFTMAX(), 1, 2); // float32=v1, uint8=v2
#else // New TF 2.4
  static tflite::MicroMutableOpResolver<8> resolver;  // NOLINT

  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddDequantize();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddPad();
  resolver.AddQuantize();
  resolver.AddReshape();
#endif

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    //error_reporter->Report("AllocateTensors() failed");
    myapp_printf("AllocateTensors() failed\r\n");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);

  // Check input format
  if (input->type != kTfLiteFloat32) {
    myapp_printf("Bad input type: %d, expected %d\r\n", input->type, kTfLiteFloat32);
  }

  if (input->dims->size != g_model_dims_len) {
    myapp_printf("Number of input dimensions doesn't match with model: %d vs %d\r\n", g_model_dims_len, input->dims->size);
  }

  input_vector_length = 1;
  for (int i = 0; i < g_model_dims_len; i++) {
    if (input->dims->data[i] != g_model_dims[i]) {
      myapp_printf("Dim n%d has length %d in model vs expected %d\r\n", i, input->dims->data[i], g_model_dims[i]);
    }
    // Compute input vector length
    input_vector_length *= g_model_dims[i];
  }

  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  //long t_start = clock(); // will overflow at one point

  //static float y_val = 0;
  // Place our calculated x value in the model's input tensor
  //for (int i = 0; i < input_vector_length; i++) {
  //  int randval = rand();
  //  input->data.f[i] = randval % 1024 / 100.0f;
  //}
  
  //myapp_printf("%s\r\n", "Wait...");
  static char buf[MAX_UART_PACKET_SIZE];
  buf[0] = '\n';
  int ret = -1;
  while (ret == -1) { // wait for input timeout — sender finished transmitting
    ret = HandleInput(MAX_UART_PACKET_SIZE, buf);
    //myapp_printf("Waiting for input %d\r\n", ret);
  }
  char *pbuf = buf;

  myapp_printf("%d\r\n", ret);

  unsigned int i = 0;
  while ((pbuf - buf) < ret && *pbuf != '\r' && *pbuf != '\n') {
    input->data.f[i] = strtof(pbuf, &pbuf);
    //myapp_printf("Parsed %d: %f, nextc: %c\r\n", i, input->data.f[i], *pbuf);
    i++;
    pbuf++;//skip delimiter
  }

  //buf[7] = '\0';
  //error_reporter->Report("%d %s", ret, buf);
  //myapp_printf("%d %s\r\n", ret, buf);

  //myapp_printf("Input: %f, Outdim: %d %d %d\r\n", input->data.f[0], output->dims->size, output->dims->data[0], output->dims->data[1]);
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
  //error_reporter->Report("Invoke failed\n" // on x_val: %f\n",
                           //static_cast<double>(x_val)
		//	   );
    myapp_printf("%s\r\n", "Invoke failed");
    return;
  }


  //int c = 0;
  //for (int i=0; i < 10000000; i++) {
  //  int a = 63*48;
  //  c += a +1 + i + ret;
  //}

  // Read the predicted y value from the model's output tensor
  //float y_val = output->data.f[0];

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  //HandleOutput(error_reporter, 0, y_val);
  //y_val = 1 - y_val;

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  //long t_stop = clock();
  //long t_diff = t_stop - t_start;
  //error_reporter->Report("out: %f %f %f %f, inference_count %d, time: %d", output->data.f[0], output->data.f[1], output->data.f[2], output->data.f[3], inference_count, t_stop);
  //myapp_printf("out: %f %f %f %f, inference_count %d, time: %d", output->data.f[0], output->data.f[1], output->data.f[2], output->data.f[3], inference_count, t_stop);
/*myapp_printf("i=%d, ic=%d, ret=%d, out0=%f, out1=%f, out2=%f, out3=%f, out4=%f, out5=%f, out6=%f, out7=%f, out8=%f, out9=%f, out10=%f, out11=%f, out12=%f, out13=%f, out14=%f, out15=%f\r\n", i, inference_count, ret,
    output->data.f[0], output->data.f[1], output->data.f[2], output->data.f[3], output->data.f[4], output->data.f[5],
    output->data.f[6], output->data.f[7], output->data.f[8], output->data.f[9], output->data.f[10], output->data.f[11],
    output->data.f[12], output->data.f[13], output->data.f[14], output->data.f[15]);
    */

  float m = output->data.f[0];
  int c = 0;
  for (int j = 1; j < output->dims->data[1]; j++) {
    if (output->data.f[j] > m) {
      m = output->data.f[j];
      c = j;
    }
  }
  myapp_printf("%d,%d,%f\r\n", inference_count, c, (double)m);

  //myapp_printf("i=%d, ic=%d, in=%f, ret=%d, out0=%f, out1=%f, out2=%f, out3=%f, out4=%f, out5=%f, out6=%f, out7=%f\r\n", i,
  //inference_count,  input->data.f[0], ret,
  //  output->data.f[0], output->data.f[1], output->data.f[2], output->data.f[3], output->data.f[4], output->data.f[5],
  //  output->data.f[6], output->data.f[7]);
}
