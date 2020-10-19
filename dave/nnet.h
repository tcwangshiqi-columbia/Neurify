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
#include <math.h>
typedef int bool;
enum { false, true };

#define NEED_OUTWARD_ROUND 0
#define OUTWARD_ROUND 0.00000005
#define MAX_PIXEL 255.0
#define MIN_PIXEL 0.0
#define MAX 1
#define MIN 0
//0.1 redius means 11.46 degree
//0.26 means 30 degree
#define DAVE_PROPERTY 0.26


extern int ERR_NODE;

extern int PROPERTY;
extern char *LOG_FILE;
extern FILE *fp;
extern float INF;
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
    int *layerTypes;   //Intermediate layer types

    // convlayersnum is the number of convolutional layers
    // convlayer is a matrix [convlayersnum][5]
    // out_channel, in_channel, kernel, stride, padding
    int convLayersNum;
    int **convLayer;
    float ****conv_matrix;
    float **conv_bias;

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
    int split_feature;

    float *out; // original output array
};

//Functions Implemented
struct NNet *load_network(const char *filename, int img);

struct NNet *load_conv_network(const char *filename, int img);

void load_inputs(int img, int inputSize, float *input);

void initialize_input_interval(struct NNet *nnet, int img, int inputSize, float *input, float *u, float *l);

/*  
 * Uses for loop to calculate the output
 * 0.0000374 sec for one run with one core
*/
int evaluate(struct NNet *network, struct Matrix *input, struct Matrix *output);

int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

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

int forward_prop_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

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
                                     struct Interval *output, float *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower,
                                     int *wrong_nodes, int *wrong_node_length,
                                     float *wrong_up_s_up, float *wrong_up_s_low,
                                     float *wrong_low_s_up, float *wrong_low_s_low);

int forward_prop_interval_equation_linear2(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower,
                                     int *wrong_nodes, int *wrong_node_length,
                                     float *wrong_up_s_up, float *wrong_up_s_low,
                                     float *wrong_low_s_up, float *wrong_low_s_low);

void sort(float *array, int num, int *ind);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num);

void add_l1_constraint(struct Interval *input, lprec *lp, int *rule_num, float l1);

void set_node_constraints(lprec *lp, float *equation, int start, int *rule_num, int sig, int inputSize);

float set_output_constraints(lprec *lp, float *equation, int start, int *rule_num, int inputSize, int is_max, float *output, float *input_prev);

float set_wrong_node_constraints(lprec *lp, float *equation, int start, int *rule_num, int inputSize, int is_max, float *output);

void destroy_network(struct NNet *network);

void destroy_conv_network(struct NNet *network);

void denormalize_input(struct NNet *nnet, struct Matrix *input);

void denormalize_input_interval(struct NNet *nnet, struct Interval *input);

void normalize_input(struct NNet *nnet, struct Matrix *input);

void normalize_input_interval(struct NNet *nnet, struct Interval *input);

void backward_prop(struct NNet *nnet, float *grad, int R[][nnet->maxLayerSize]);

void backward_prop_conv(struct NNet *nnet, float *grad, int R[][nnet->maxLayerSize]);

void backward_prop_old(struct NNet *nnet, struct Interval *grad, int R[][nnet->maxLayerSize]);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num);

// for one shot approximation
int forward_prop_interval_equation_linear_conv(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *grad,
                                     float *equation, float *equation_err,
                                     float *new_equation, float *new_equation_err,
                                     int *wrong_nodes, int *wrong_node_length,
                                     int *full_wrong_node_length,
                                     float *equation_conv, float *equation_conv_err, int *err_row_conv);

// for approximation with input split
int forward_prop_input_interval_equation_linear_conv(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *equation, float *equation_err,
                                     float *new_equation, float *new_equation_err);
