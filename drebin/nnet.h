/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang
 ** This file is part of the Neurify project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include "matrix.h"
#include <string.h>
#include "interval.h"
#include "lp_dev/lp_lib.h"
#include <time.h>
typedef int bool;
enum { false, true };

#define NEED_OUTWARD_ROUND 0
#define OUTWARD_ROUND 0.00000005
#define MAX_PIXEL 255.0
#define MIN_PIXEL 0.0
#define MAX 1
#define MIN 0

extern int PROPERTY;
extern char *LOG_FILE;
extern FILE *fp;
extern int INF;
extern int ACCURATE_MODE;
extern struct timeval start,finish, last_finish;

//Neural Network Struct
struct NNet 
{
    int symmetric;     //1 if network is symmetric, 0 otherwise
    int numLayers;     //Number of layers in the network
    int inputSize;     //Number of inputs to the network
    int outputSize;    //Number of outputs to the network
    int maxLayerSize;  //Maximum size dimension of a layer in the network
    int *layerSizes;   //Array of the dimensions of the layers in the network

    float min;      //Minimum value of inputs
    float max;     //Maximum value of inputs
    float mean;     //Array of the means used to scale the inputs and outputs
    float range;    //Array of the ranges used to scale the inputs and outputs
    float ****matrix; //4D jagged array that stores the weights and biases
                       //the neural network.
    struct Matrix* weights;
    struct Matrix* bias;
    struct Matrix* pos_weights;
    struct Matrix* neg_weights;

    int target;
    int *feature_range;
    int feature_range_length;
};

//Functions Implemented
struct NNet *load_network(const char *filename, struct Matrix *input_prev_matrix);

int load_inputs_drebin(int app, float *input);

void initialize_input_interval(struct NNet *nnet, int img, int inputSize, float *input, float *u, float *l);

/*  
 * Uses for loop to calculate the output
 * 0.0000374 sec for one run with one core
*/
int evaluate(struct NNet *network, struct Matrix *input, struct Matrix *output);
/*  
 * Uses for loop to calculate the interval output
 * 0.000091 sec for one run with one core
*/
int evaluate_interval(struct NNet *network, struct Interval *input, struct Interval *output);
/*  
 * Uses for loop with equation to calculate the interval output
 * 0.000229 sec for one run with one core
*/
int evaluate_interval_equation(struct NNet *network, struct Interval *input, struct Interval *output);

/*  
 * Uses sgemm to calculate the output
 * 0.0000218 sec for one run with one core
*/
int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output);

int forward_prop_interval(struct NNet *network, struct Interval *input, struct Interval *output);
/*  
 * Uses sgemm with equation to calculate the interval output
 * 0.000857 sec for one run with one core
*/
int forward_prop_interval_equation(struct NNet *network, struct Interval *input,
                                     struct Interval *output, struct Interval *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower);


int forward_prop_interval_equation2(struct NNet *network, struct Interval *input,
                                     struct Interval *output, struct Interval *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower);

int forward_prop_interval_equation_linear(struct NNet *network, struct Interval *input,
                                     struct Interval *output, struct Interval *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower);

int forward_prop_interval_equation_linear2(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower,
                                     int *wrong_nodes, int *wrong_node_length,
                                     float *wrong_up_s_up, float *wrong_up_s_low,
                                     float *wrong_low_s_up, float *wrong_low_s_low);

void sort(float *array, int num, int *ind);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num);

void set_node_constraints(lprec *lp, float *equation, int start, int *rule_num, int sig, int inputSize);

float set_output_constraints(lprec *lp, float *equation, int start, int *rule_num, int inputSize, int is_max, float *output, float *input_prev);

float set_wrong_node_constraints(lprec *lp, float *equation, int start, int *rule_num, int inputSize, int is_max, float *output);

void destroy_network(struct NNet *network);

void denormalize_input(struct NNet *nnet, struct Matrix *input);

void denormalize_input_interval(struct NNet *nnet, struct Interval *input);

void normalize_input(struct NNet *nnet, struct Matrix *input);

void normalize_input_interval(struct NNet *nnet, struct Interval *input);
/*
 * The back prop to calculate the gradient
 * 0.000249 sec for one run with one core
*/
void backward_prop(struct NNet *nnet, float *grad, int R[][nnet->maxLayerSize]);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num);