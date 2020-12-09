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


extern int ERR_NODE;

extern int PROPERTY;
extern char *LOG_FILE;
extern FILE *fp;
extern float INF;
extern int NORM_INPUT;
extern struct timeval start,finish, last_finish;

//Neural Network Struct
struct NNet 
{
    int numLayers;     //Number of layers in the network
    int inputSize;     //Number of inputs to the network
    int outputSize;    //Number of outputs to the network
    int maxLayerSize;  //Maximum size dimension of a layer in the network
    int *layerSizes;   //Array of the dimensions of the layers in the network
    int *layerTypes;   //Intermediate layer types

    /*
     * convlayersnum is the number of convolutional layers
     * convlayer is a matrix [convlayersnum][5]
     * out_channel, in_channel, kernel, stride, padding
    */
    int convLayersNum;
    int **convLayer;
    float ****conv_matrix;
    float **conv_bias;

    float min;      //Minimum value of inputs
    float max;     //Maximum value of inputs
    float ****matrix; //4D jagged array that stores the weights and biases
                       //the neural network.
    struct Matrix* weights;
    struct Matrix* bias;

    int target;
    int *feature_range;
    int feature_range_length;
    int split_feature;
};


struct SymInterval
{
    struct Matrix *eq_matrix;
    struct Matrix *err_matrix;
};


void sym_fc_layer(struct SymInterval *sInterval, struct SymInterval *new_sInterval, struct NNet *nnet, int layer, int err_row);


void sym_conv_layer(struct SymInterval *sInterval, struct SymInterval *new_sInterval, struct NNet *nnet, int layer, int err_row);

void relu_bound(struct SymInterval *sInterval, struct NNet *nnet, 
                struct Interval *input, int i, int layer, int err_row, 
                float *low, float *up);

int relax_relu(struct NNet *nnet, struct SymInterval *sym_interval,
    float lower_bound, float upper_bound, int i,
    int err_row, int *wrong_node_length, int *wcnt);

int sym_relu_layer(struct SymInterval *new_sInterval, struct Interval *input, struct Interval *output,
                    struct NNet *nnet, int R[][nnet->maxLayerSize],
                    int layer, int err_row,
                    int *wrong_nodes, int * wrong_node_length, int *node_cnt);

//Functions Implemented
struct NNet *load_conv_network(const char *filename, float *input);

void load_inputs(int img, int inputSize, float *input);

int find_in_h(int needle, int* h, int h_size);
void initialize_input_interval(struct NNet *nnet, float *input, int input_size, int *h, int h_size, float *u, float *l, float *u_bounds, float *l_bounds);

/*  
 * Uses for loop to calculate the output
 * 0.0000374 sec for one run with one core
*/
int evaluate(struct NNet *network, struct Matrix *input, struct Matrix *output);

int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

/*  
 * Uses sgemm to calculate the output
 * 0.0000218 sec for one run with one core
*/
int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output);

void forward_prop_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

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

void sort_layers(int numLayers, int*layerSizes, int wrong_node_length, int*wrong_nodes);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num, int inputSize);

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
/*
 * The back prop to calculate the gradient
 * 0.000249 sec for one run with one core
*/
void backward_prop(struct NNet *nnet, float *grad, int R[][nnet->maxLayerSize]);

void backward_prop_conv(struct NNet *nnet, float *grad, int R[][nnet->maxLayerSize]);

void backward_prop_old(struct NNet *nnet, struct Interval *grad, int R[][nnet->maxLayerSize]);

/*
int forward_prop_interval_equation_linear_try(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *grad,
                                     float *equation, float *equation_err,
                                     float *new_equation, float *new_equation_err,
                                     int *wrong_nodes, int *wrong_node_length,
                                     float *wrong_up_s_up, float *wrong_up_s_low,
                                     float *wrong_low_s_up, float *wrong_low_s_low);
*/
void forward_prop_interval_equation_linear_conv(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *grad,
                                     int *wrong_nodes, int *wrong_node_length,
                                     int *full_wrong_node_length,
                                     float *equation_conv, float *equation_conv_err, int *err_row_conv);
