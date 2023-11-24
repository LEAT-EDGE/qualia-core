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

#include "../input_handler.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "am_bsp.h"  // NOLINT
#include "am_util_stdio.h"

#define UART_IDENT 0
#define READ_BLOCK_SIZE 32
void *g_pvUART;
uint8_t g_pui8UARTTXBuffer[MAX_UART_PACKET_SIZE];
uint8_t g_psWriteData[MAX_UART_PACKET_SIZE];
static char g_prfbuf[AM_PRINTF_BUFSIZE];
volatile uint32_t g_ui32UARTRxIndex = 0;
volatile bool g_bRxTimeoutFlag = false;


//FIXME:delete
typedef struct
{
    bool bValid;
    uint32_t regILPR;
    uint32_t regIBRD;
    uint32_t regFBRD;
    uint32_t regLCRH;
    uint32_t regCR;
    uint32_t regIFLS;
    uint32_t regIER;
}
am_hal_uart_register_state_t;
typedef struct
{
    am_hal_handle_prefix_t prefix;
    am_hal_uart_register_state_t sRegState;
    
    uint32_t ui32Module;

    bool bEnableTxQueue;
    am_hal_queue_t sTxQueue;

    bool bEnableRxQueue;
    am_hal_queue_t sRxQueue;

    uint32_t ui32BaudRate;
}
am_hal_uart_state_t;
//FIXME



int HandleInput(unsigned int size, char *buf) {
  /*if (!g_bRxTimeoutFlag) {
    return -1;
  }*/

  if (g_bRxTimeoutFlag) {
    g_bRxTimeoutFlag = false;
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + UART_IDENT));
  }

  //myapp_printf("lastchar %c %d\r\n", g_psWriteData[g_ui32UARTRxIndex - 1], g_psWriteData[g_ui32UARTRxIndex - 1]);

  if (g_ui32UARTRxIndex < 1 || g_psWriteData[g_ui32UARTRxIndex - 1] != '\n') {
    // cannot rely only on timeout if sender is too slow, so check EOL, also make sure we got some data
    return -1;
  }

  if (g_ui32UARTRxIndex < size) {
    size = g_ui32UARTRxIndex;
  }

  for (unsigned int i = 0; i < size; i++) {
    buf[i] = g_psWriteData[i];
  }

  g_ui32UARTRxIndex = 0;

  return size;
}

void myapp_uart_init(void) {
  //
  // Start the UART.
  //
  am_hal_uart_config_t sUartConfig =
  {
    //
    // Standard UART settings: 115200-8-N-1
    //
    //.ui32BaudRate    = 115200,
    .ui32BaudRate    = 921600,
    .ui32DataBits    = AM_HAL_UART_DATA_BITS_8,
    .ui32Parity      = AM_HAL_UART_PARITY_NONE,
    .ui32StopBits    = AM_HAL_UART_ONE_STOP_BIT,
    .ui32FlowControl = AM_HAL_UART_FLOW_CTRL_NONE,

    //
    // Set TX and RX FIFOs to interrupt at three-quarters full.
    //
    .ui32FifoLevels = (AM_HAL_UART_TX_FIFO_1_4 |
                       AM_HAL_UART_RX_FIFO_1_4),

    //
    // This code will use the standard interrupt handling for UART TX, but
    // we will have a custom routine for UART RX.
    //
    .pui8TxBuffer = g_pui8UARTTXBuffer,
    .ui32TxBufferSize = sizeof(g_pui8UARTTXBuffer),
    .pui8RxBuffer = 0,
    .ui32RxBufferSize = 0,
  };

  am_hal_uart_initialize(UART_IDENT, &g_pvUART);
  am_hal_uart_power_control(g_pvUART, AM_HAL_SYSCTRL_WAKE, false);
  am_hal_uart_configure(g_pvUART, &sUartConfig);
  am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
  am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);

  //
  // Make sure to enable the interrupts for RX, since the HAL doesn't already
  // know we intend to use them.
  //
  NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + UART_IDENT));
  am_hal_uart_interrupt_enable(g_pvUART, (AM_HAL_UART_INT_RX |
                               AM_HAL_UART_INT_RX_TMOUT));

  am_hal_interrupt_master_enable();


  //am_util_stdio_printf_init(myapp_uart_string_print);
}

//*****************************************************************************
//
// Interrupt handler for the UART.
//
//*****************************************************************************
extern "C" void am_uart_isr(void)
{
  uint32_t ui32Status;

  //
  // Read the masked interrupt status from the UART.
  //
  am_hal_uart_interrupt_status_get(g_pvUART, &ui32Status, true);
  am_hal_uart_interrupt_clear(g_pvUART, ui32Status);
  am_hal_uart_interrupt_service(g_pvUART, ui32Status, 0);


  //
  // If there's an RX interrupt, handle it in a way that preserves the
  // timeout interrupt on gaps between packets.
  //
  if (ui32Status & (AM_HAL_UART_INT_RX_TMOUT | AM_HAL_UART_INT_RX))
  {
      uint32_t ui32BytesRead = 0;
    if (g_ui32UARTRxIndex + READ_BLOCK_SIZE <= MAX_UART_PACKET_SIZE) { // buffer has free space

      am_hal_uart_transfer_t sRead =
      {
        .ui32Direction = AM_HAL_UART_READ,
        .pui8Data = (uint8_t *) &(g_psWriteData[g_ui32UARTRxIndex]),
        //.pui8Data = (uint8_t *) &(g_psWriteData[0]),
        .ui32NumBytes = READ_BLOCK_SIZE,
        .ui32TimeoutMs = 0,
        .pui32BytesTransferred = &ui32BytesRead,
        //.pui32BytesTransferred = NULL,
      };

      am_hal_uart_transfer(g_pvUART, &sRead);
      //ui32BytesRead=0;

      g_ui32UARTRxIndex += ui32BytesRead;
    }
    //myapp_printf("%d %d %d %d\r\n", g_ui32UARTRxIndex, READ_BLOCK_SIZE, MAX_UART_PACKET_SIZE, ui32BytesRead);

    //
    // If there is a TMOUT interrupt, assume we have a compete packet, and
    // send it over SPI.
    //
    if (ui32Status & (AM_HAL_UART_INT_RX_TMOUT))
    {
      NVIC_DisableIRQ((IRQn_Type)(UART0_IRQn + UART_IDENT));
      //cmd_handler(g_psWriteData, g_ui32UARTRxIndex);
      g_bRxTimeoutFlag = true;
    }
  }

}

//*****************************************************************************
//
//! @brief UART-based string print function.
//!
//! This function is used for printing a string via the UART, which for some
//! MCU devices may be multi-module.
//!
//! @return None.
//
//*****************************************************************************
void myapp_uart_string_print(char *pcString) {
    uint32_t ui32StrLen = 0;
    uint32_t ui32BytesWritten = 0;

    //
    // Measure the length of the string.
    //
    while (pcString[ui32StrLen] != 0)
    {
        ui32StrLen++;
    }

    //
    // Print the string via the UART.
    //
    const am_hal_uart_transfer_t sUartWrite =
    {
        .ui32Direction = AM_HAL_UART_WRITE,
        .pui8Data = (uint8_t *) pcString,
        .ui32NumBytes = ui32StrLen,
        .ui32TimeoutMs = AM_HAL_UART_WAIT_FOREVER,
        .pui32BytesTransferred = &ui32BytesWritten,
    };

    am_hal_uart_transfer(g_pvUART, &sUartWrite);

#if 0
    if (ui32BytesWritten != ui32StrLen)
    {
        //
        // Couldn't send the whole string!!
        //
        while(1);
    }
#endif
} // am_bsp_uart_string_print()

uint32_t myapp_printf(const char *pcFmt, ...) {
  uint32_t ui32NumChars;

  //
  // Convert to the desired string.
  //
  va_list pArgs;
  va_start(pArgs, pcFmt);
  ui32NumChars = am_util_stdio_vsprintf(g_prfbuf, pcFmt, pArgs);
  va_end(pArgs);

  //
  // This is where we print the buffer to the configured interface. 
  //
  myapp_uart_string_print(g_prfbuf);

  //
  // return the number of characters printed.
  //
  return ui32NumChars; 
}


int AMMicroErrorReporter::Report(const char* format, va_list args) {                                                                  
  uint32_t ui32NumChars;
  ui32NumChars = am_util_stdio_vsprintf(g_prfbuf, format, args);
  myapp_uart_string_print(g_prfbuf);
  return ui32NumChars;
}


void enable_burst_mode(void) {
  am_hal_burst_avail_e          eBurstModeAvailable;
  am_hal_burst_mode_e           eBurstMode;

  //
  // Check that the Burst Feature is available.
  //
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_initialize(&eBurstModeAvailable)) {
#if 0 //DEBUG
    if (AM_HAL_BURST_AVAIL == eBurstModeAvailable) {
      myapp_printf("Apollo3 Burst Mode is Available\n");
    } else {
      myapp_printf("Apollo3 Burst Mode is Not Available\n");
    }
  } else {
    myapp_printf("Failed to Initialize for Burst Mode operation\n");
#endif
  }

  //
  // Put the MCU into "Burst" mode.
  //
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode)) {
#if 0 //DEBUG
    if (AM_HAL_BURST_MODE == eBurstMode) {
      myapp_printf("Apollo3 operating in Burst Mode (96MHz)\n");
    }
  } else {
    myapp_printf("Failed to Enable Burst Mode operation\n");
#endif
  }
}
