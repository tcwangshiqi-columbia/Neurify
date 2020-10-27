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
extern bool adv_found;
extern bool analysis_uncertain;

extern int CHECK_ADV_MODE;

extern int thread_count;

extern int MAX_DEPTH;

extern int progress;

extern int progress_list[PROGRESS_DEPTH];
extern int total_progress[PROGRESS_DEPTH];


#define ADV_THRESHOLD  0.00001


pthread_mutex_t lock;
extern int count;



struct direct_run_check_conv_lp_args
{
	struct NNet *nnet;
	struct Interval *input;
	bool *output_map;
	float *grad;
	int *sigs;
	float *equation_conv;
	float *equation_conv_err;
	float err_row_conv;
	int target;
	lprec *lp;
	int *rule_num;
	int depth;
};



void check_adv(struct NNet *nnet, struct Interval *input);
void check_adv1(struct NNet* nnet, struct Matrix *adv);
int check_l1(float *input, int inputSize);

int check_functions(struct NNet *nnet, struct Interval *output);

int check_functions_norm(struct NNet *nnet, struct Interval *output);

int check_functions1(struct NNet *nnet, struct Matrix *output);

int check_not_min(struct NNet *nnet, struct Interval *output);

int check_not_min1(struct NNet *nnet, struct Matrix *output);

int check_not_min_p7(struct NNet *nnet, struct Interval *output);

int check_not_min1_p7(struct NNet *nnet, struct Matrix *output);

int pop_queue(int *wrong_nodes, int *wrong_node_length);

int search_queue(int *wrong_nodes, int *wrong_node_length, int node_cnt);


bool forward_prop_interval_equation_conv_lp(struct NNet *nnet,
	struct Interval *input, bool *output_map, int *wrong_nodes,
	int *wrong_node_length, int *sigs, float *equation_conv,
	float *equation_conv_err, float err_row_conv, int target, lprec *lp,
	int *rule_num);

bool direct_run_check_conv_lp(struct NNet *nnet, struct Interval *input, bool *output_map, float *grad,
                     int *sigs,
                     float *equation_conv, float *equation_conv_err, float err_row_conv,
                     int target, lprec *lp, int *rule_num, int depth);

bool split_interval_conv_lp(struct NNet *nnet, struct Interval *input, bool *output_map, float *grad,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *equation_conv, float *equation_conv_err, float err_row_conv,
                     lprec *lp, int *rule_num, int depth);
