#ifndef INPUT_HANDLER_H_
#define INPUT_HANDLER_H_

#include <stdint.h>

#include "tensorflow/lite/core/api/error_reporter.h"

#define MAX_UART_PACKET_SIZE            24576 // Must be able to hold at least one full message of max length

class AMMicroErrorReporter : public tflite::ErrorReporter {
 public:
  ~AMMicroErrorReporter() {}
  int Report(const char* format, va_list args) override;
};

int HandleInput(unsigned int size, char *buf);
void myapp_uart_init(void);
uint32_t myapp_printf(const char *pcFmt, ...);
void myapp_uart_string_print(char *pcString);

void enable_burst_mode(void);

#endif  // INPUT_HANDLER_H_
