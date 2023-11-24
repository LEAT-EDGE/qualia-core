MYAPP_SRCS := \
main.cc \
main_functions.cc \
model_data.cc \
output_handler.cc \
input_handler.cc \

MYAPP_HDRS := \
model_data.h \
output_handler.h \
input_handler.h \
main_functions.h

#Find any platform-specific rules for this example.
include $(wildcard myapp/*/Makefile.inc)


PROJECT_PATH := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

MYAPP_TEST_SRCS := $(addprefix $(PROJECT_PATH)/,$(MYAPP_TEST_SRCS))
MYAPP_TEST_HDRS := $(addprefix $(PROJECT_PATH)/,$(MYAPP_TEST_HDRS))
OUTPUT_HANDLER_TEST_SRCS := $(addprefix $(PROJECT_PATH)/,$(OUTPUT_HANDLER_TEST_SRCS))
OUTPUT_HANDLER_TEST_HDRS := $(addprefix $(PROJECT_PATH)/,$(OUTPUT_HANDLER_TEST_HDRS))
MYAPP_SRCS := $(addprefix $(PROJECT_PATH)/,$(MYAPP_SRCS))
MYAPP_HDRS := $(addprefix $(PROJECT_PATH)/,$(MYAPP_HDRS))

# Tests loading and running the sine model.
#$(eval $(call microlite_test,myapp_test,\
#$(MYAPP_TEST_SRCS),$(MYAPP_TEST_HDRS)))

# Tests producing an output.
#$(eval $(call microlite_test,output_handler_test,\
#$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,myapp,\
$(MYAPP_SRCS),$(MYAPP_HDRS)))
