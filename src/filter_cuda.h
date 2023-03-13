#include "utils.h"
#include <math.h>
#include "gif_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

void apply_gray_filter_gpu(animated_gif *image);
void apply_sobel_filter_gpu(animated_gif *image);
void apply_all_filters_gpu(animated_gif *image);

#ifdef __cplusplus
}
#endif