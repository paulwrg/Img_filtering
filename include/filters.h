#ifndef FILTERS_H
#define FILTERS_H

#include "utils.h"


#define THREAD_NUM 16

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

void apply_gray_filter( animated_gif * image , int image_index);
void apply_blur_filter( animated_gif * image, int size, int threshold , int image_index);
void apply_sobel_filter( animated_gif * image , int image_index);
void apply_all_filters(animated_gif* image);

#endif /* FILTERS_H */