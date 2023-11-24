#include "model_data.h"

// We need to keep the data array aligned on some architectures.
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif


const unsigned char g_model_data[] DATA_ALIGN_ATTRIBUTE = {
	${data}
};
const int g_model_data_len = ${datalen};
const int g_model_dims[] = {${dims}};
const int g_model_dims_len = ${dimslen};
