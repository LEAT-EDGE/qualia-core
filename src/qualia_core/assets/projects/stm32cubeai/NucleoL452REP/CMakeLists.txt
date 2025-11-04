cmake_minimum_required(VERSION 3.13)

project(STM32CubeAI-NucleoL452REP C CXX ASM)

set(NETWORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model" CACHE PATH "Path to X-CUBE-AI generated code")

set(CMAKE_SYSTEM_PROCESSOR arm)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

SET(CMAKE_ASM_FLAGS "-x assembler-with-cpp")

# here is the target environment located
#set(CMAKE_FIND_ROOT_PATH  )

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

## WARNING: do not change optimization level! -O1 and higher do not work with X-CUBE-AI
set(NucleoL452REP_CFLAGS
  "-mcpu=cortex-m4"
  "-mfpu=fpv4-sp-d16"
  "-mfloat-abi=hard"
  "-mthumb"

  "-Wall"
  "-Wextra"
  "-Werror=double-promotion"

  "-fdata-sections"
  "-ffunction-sections"

   $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
   $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>

  "-O0"
)

add_compile_options(
  ${NucleoL452REP_CFLAGS}
)

add_link_options(
  ${NucleoL452REP_CFLAGS}

  "-specs=nano.specs"
  "-T" "${CMAKE_CURRENT_SOURCE_DIR}/stm32l452retxp_flash.ld"
  "-Wl,--wrap=malloc"
  "-Wl,--wrap=free" 
  "-u" "_printf_float"
  "-Wl,--gc-sections"
  "-z" "separate-code"
  "-Wl,--print-memory-usage"
)

add_compile_definitions(
  "STM32L452xx"
)

add_library(network
  ${NETWORK_DIR}/network_data_params.c
  ${NETWORK_DIR}/network.c
  ${NETWORK_DIR}/network_data.c
)

target_include_directories(network PUBLIC
  #X-CUBE-AI/App
  Middlewares/ST/AI/Inc
  ${NETWORK_DIR}
)

add_library(x-cube-ai
  X-CUBE-AI/App/aiPbIO.c
  X-CUBE-AI/App/aiValidation.c
  X-CUBE-AI/App/pb_decode.c
  X-CUBE-AI/App/aiPbMgr.c
  X-CUBE-AI/App/stm32msg.pb.c
  X-CUBE-AI/App/pb_common.c
  X-CUBE-AI/App/aiPbMemRWServices.c
  X-CUBE-AI/App/pb_encode.c
  X-CUBE-AI/App/ai_device_adaptor.c
  X-CUBE-AI/App/syscalls.c
  X-CUBE-AI/App/aiTestUtility.c
  X-CUBE-AI/App/lc_print.c
  X-CUBE-AI/App/aiTestHelper.c
  X-CUBE-AI/App/app_x-cube-ai.c
)

target_include_directories(x-cube-ai PUBLIC
  Core/Inc
  Drivers/CMSIS/Device/ST/STM32L4xx/Include
  Drivers/CMSIS/Include
  Drivers/STM32L4xx_HAL_Driver/Inc
  Middlewares/ST/AI/Inc
  X-CUBE-AI
  X-CUBE-AI/App
  X-CUBE-AI/Target
)

target_link_libraries(x-cube-ai PUBLIC
  network
)

add_executable(${PROJECT_NAME}
  Core/Src/main.c
  Core/Src/stm32l4xx_it.c
  Core/Src/stm32l4xx_hal_msp.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_crc.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_crc_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_uart.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_uart_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_rcc.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_rcc_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash_ramfunc.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_gpio.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_i2c.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_i2c_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_dma.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_dma_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_cortex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_exti.c
  Core/Src/system_stm32l4xx.c
  startup_stm32l452xx.s
)

target_include_directories(${PROJECT_NAME} PUBLIC
  Core/Inc
  Drivers/STM32L4xx_HAL_Driver/Inc
  Drivers/STM32L4xx_HAL_Driver/Inc/Legacy
  Drivers/CMSIS/Device/ST/STM32L4xx/Include
  Drivers/CMSIS/Include
)

target_link_directories(${PROJECT_NAME} PUBLIC
  Middlewares/ST/AI/Lib
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  x-cube-ai
  nosys
  :NetworkRuntime810_CM4_GCC.a
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
  "USE_HAL_DRIVER"
)
