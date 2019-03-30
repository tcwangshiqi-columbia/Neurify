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


#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include<signal.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "nnet.h"

#define PROGRESS_DEPTH 12

extern int NEED_PRINT;
extern int NEED_FOR_ONE_RUN;
extern int input_depth;
extern int adv_found;
extern int can_t_prove;

extern int CHECK_ADV_MODE;
extern int PARTIAL_MODE;

extern int thread_count;
extern float max_depth;
extern int leaf_num;

extern int progress;

extern int progress_list[PROGRESS_DEPTH];
extern int total_progress[PROGRESS_DEPTH];


#define ADV_THRESHOLD  0.00001


pthread_mutex_t lock;
extern int count;


struct direct_run_check_args
{
	struct NNet *nnet;
	struct Interval *input;
	struct Interval *output;
	struct Interval *grad; 
	int depth;
	int *feature_range;
	int feature_range_length;
	int split_feature;
	float *equation_upper;
	float *equation_lower;
	float *new_equation_upper;
	float *new_equation_lower;
	//int avg_depth;
};

struct direct_run_check_lp_args
{
	struct NNet *nnet;
	struct Interval *input;
	struct Interval *grad; 
	int *output_map;
	float *equation_upper;
	float *equation_lower;
	float *new_equation_upper;
	float *new_equation_lower;
	int *wrong_nodes;
	int *wrong_node_length;
	int *sigs;
	float *wrong_up_s_up;
	float *wrong_up_s_low;
	float *wrong_low_s_up;
	float *wrong_low_s_low;
	int target;
	int sig;
	lprec *lp;
	int *rule_num;
	int depth;
};

struct direct_run_check_conv_lp_args
{
	struct NNet *nnet;
	struct Interval *input;
	int *output_map;
	float *equation;
	float *equation_err;
	float *new_equation;
	float *new_equation_err;
	int *wrong_nodes;
	int *wrong_node_length;
	int *sigs;
	float *equation_conv;
	float *equation_conv_err;
	float err_row_conv;
	int target;
	int sig;
	lprec *lp;
	int *rule_num;
	int depth;
};



void check_adv(struct NNet *nnet, struct Interval *input);

int check_functions(struct NNet *nnet, struct Interval *output);

int check_functions_norm(struct NNet *nnet, struct Interval *output);

int check_functions1(struct NNet *nnet, struct Matrix *output);

int check_not_min(struct NNet *nnet, struct Interval *output);

int check_not_min1(struct NNet *nnet, struct Matrix *output);

int check_not_min_p7(struct NNet *nnet, struct Interval *output);

int check_not_min1_p7(struct NNet *nnet, struct Matrix *output);

void *direct_run_check_thread(void *args);

int direct_run_check(struct NNet *nnet, struct Interval *input,
				 struct Interval *output, struct Interval *grad, 
				 int depth, int *feature_range, int feature_range_length, 
				 int split_feature, float *equation_upper, float *equation_lower,
				 float *new_equation_upper, float *new_equation_lower);


int pop_queue(int *wrong_nodes, int *wrong_node_length);

int search_queue(int *wrong_nodes, int *wrong_node_length, int node_cnt);


int split_interval(struct NNet *nnet, struct Interval *input,
			   struct Interval *output, struct Interval *grad, 
			   int depth, int *feature_range, int feature_range_length, 
			   int split_feature, float *equation_upper, float *equation_lower,
			   float *new_equation_upper, float *new_equation_lower);

int forward_prop_interval_equation_lp(struct NNet *nnet, struct Interval *input,
                                 struct Interval *grad, int *output_map,
                                 float *equation_upper, float *equation_lower,
                                 float *new_equation_upper, float *new_equation_lower,
                                 int *wrong_nodes, int *wrong_node_length, int *sigs,
                                 float *wrong_up_s_up, float *wrong_up_s_low,
                                 float *wrong_low_s_up, float *wrong_low_s_low,
                                 int target, int sig,
                                 lprec *lp, int *rule_num);


int forward_prop_interval_equation_conv_lp(struct NNet *nnet, struct Interval *input, int *output_map,
                                 float *equation, float *equation_err,
                                 float *new_equation, float *new_equation_err, int *sigs,
                                 float *equation_conv, float *equation_conv_err, float err_row_conv,
                                 int target, int sig,
                                 lprec *lp, int *rule_num);

int direct_run_check_lp(struct NNet *nnet, struct Interval *input,
                     struct Interval *grad, int *output_map,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *wrong_up_s_up, float *wrong_up_s_low,
                     float *wrong_low_s_up, float *wrong_low_s_low,
                     int target, int sig,
                     lprec *lp, int *rule_num, int depth);

int direct_run_check_conv_lp(struct NNet *nnet, struct Interval *input, int *output_map,
                     float *equation, float *equation_err,
                     float *new_equation, float *new_equation_err,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *equation_conv, float *equation_conv_err, float err_row_conv,
                     int target, int sig,
                     lprec *lp, int *rule_num, int depth);



int split_interval_lp(struct NNet *nnet, struct Interval *input,
                     struct Interval *grad, int *output_map,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *wrong_up_s_up, float *wrong_up_s_low,
                     float *wrong_low_s_up, float *wrong_low_s_low,
                     lprec *lp, int *rule_num, int depth);

int split_interval_conv_lp(struct NNet *nnet, struct Interval *input, int *output_map,
                     float *equation, float *equation_err,
                     float *new_equation, float *new_equation_err,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *equation_conv, float *equation_conv_err, float err_row_conv,
                     lprec *lp, int *rule_num, int depth);
