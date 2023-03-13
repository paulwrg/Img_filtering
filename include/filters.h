#ifndef FILTERS_H
#define FILTERS_H

#include "utils.h"


#define THREAD_NUM 1

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

void apply_gray_filter(animated_gif * image, int image_index);
void apply_blur_filter(animated_gif * image, int size, int threshold, int image_index);
void apply_sobel_filter(animated_gif * image, int image_index);
void apply_all_filters(animated_gif* image);

// TODO add support for horizontal splitting too
void apply_sobel_filter_with_splitting(animated_gif * image, int image_index, int start, int stop);
void apply_blur_filter_with_splitting(animated_gif * image, int size, int threshold, int image_index, int start, int stop);
void apply_sobel_filter_with_splitting(animated_gif * image, int image_index, int start, int stop);

#endif /* FILTERS_H */