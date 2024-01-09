//
// Created by mat on 30/10/23.
//

#ifndef S3PROJECT_SETTINGS_H
#define S3PROJECT_SETTINGS_H
# define M_PI           3.14159265358979323846  /* pi */
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#if USE_SSE2 && USE_AVX2
#error "Cannot use both SSE2 and AVX2"
#endif

#define NUM_FILTERS 32
#define FILTERS_SIZE 3

#define POOL_FILTERS_SIZE 2
#define POOL_STRIDE 2

#define WINW 700
#define WINH 700

#define SAVE_FOLD "out"

#define BIL_FIL_DIAM 3
#define BIL_FIL_SIG_COL 3
#define BIL_FIL_SIG_SPACE 3

#define ADAPTIVE_THRESH_BLOCK_SIZE 11
#define ADAPTIVE_THRESH_STRENGTH 3

#define ADAPTIVE_THRESH2_THRESHOLD 0.55f

#define ADAPTIVE_THRESHS_HIGH 255
#define ADAPTIVE_THRESHS_LOW 0

#define DELATE_KERNEL_SIZE 3

#define SQUARES_EDGE_DIFF_TOL .8f

#define GAUSS_BLUR_SIGMA 1.0
#define GAUSS_BLUR_K 1

#define MIN_LINES_RHO_DIST 40
#define MIN_LINES_THETA_DIST 1.04719758033f // 60 degrees

#define MIN_LINES 150
#define MAX_LINES 200

#define MIN_INTERSECTION_DIST 30

#define MAX_RIGHT_ANGLE_ABS_DOT_PRODUCT .1f

#define SEGMENT_BLACK_PERCENTAGE_NUM_SAMPLES 200

#define BORDER_CROP_PERCENTAGE 10
#define EMPTY_CELL_THRESHOLD .04f

#define VERBOSE 1
#define MULTITHREAD 1

#if MULTITHREAD
#define NUM_THREADS sysconf(_SC_NPROCESSORS_ONLN)
#else
#define NUM_THREADS 1
#endif

// ANSI escape codes for text colors
#define BLACK "\033[0;30m"
#define RED "\033[0;31m"
#define GREEN "\033[0;32m"
#define YELLOW "\033[0;33m"
#define BLUE "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN "\033[0;36m"
#define WHITE "\033[0;37m"

// ANSI escape codes for bold text colors
#define BOLD_BLACK "\033[1;30m"
#define BOLD_RED "\033[1;31m"
#define BOLD_GREEN "\033[1;32m"
#define BOLD_YELLOW "\033[1;33m"
#define BOLD_BLUE "\033[1;34m"
#define BOLD_MAGENTA "\033[1;35m"
#define BOLD_CYAN "\033[1;36m"
#define BOLD_WHITE "\033[1;37m"

// ANSI escape codes for background colors
#define BG_BLACK "\033[40m"
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN "\033[46m"
#define BG_WHITE "\033[47m"

// ANSI escape code to reset text color and background color
#define RESET "\033[0m"
#endif //S3PROJECT_SETTINGS_H
