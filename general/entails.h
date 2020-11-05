#include <openblas/cblas.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
void sig_handler(int signo);
int entails( int *h, int h_size, float *input, int input_size, float eps, char* network_path);
