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

#include "split.h"

#define AVG_WINDOW 5
#define MAX_THREAD 56
#define MIN_DEPTH_PER_THREAD 5 

int NEED_PRINT = 0;
int NEED_FOR_ONE_RUN = 0;
int input_depth = 0;
int adv_found = 0;
int can_t_prove = 0;
int count = 0;
int thread_tot_cnt  = 0;
int smear_cnt = 0;

int progress = 0;

int CHECK_ADV_MODE = 0;
int PARTIAL_MODE = 0;

float avg_depth = 50;
float total_avg_depth = 0;
int leaf_num = 0;
float max_depth = 0;
int progress_list[PROGRESS_DEPTH];
int total_progress[PROGRESS_DEPTH];

int check_not_max(struct NNet *nnet, struct Interval *output, int normalize_output){
    if(!normalize_output){
        for(int i=0;i<nnet->outputSize;i++){
            if(output->upper_matrix.data[i]-output->lower_matrix.data[nnet->target]>0 && i != nnet->target){
                return 1;
            }
        }
        return 0;
    }
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]>0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}


int check_max_constant(struct NNet *nnet, struct Interval *output){
    if(output->upper_matrix.data[nnet->target]>0.5011){
        return 1;
    }
    else{
        return 0;
    }
}

int check_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->lower_matrix.data[i]>0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_min(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]<0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_min_p7(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i != 3 && i != 4){
            if(output->upper_matrix.data[i]< output->lower_matrix.data[4] && output->upper_matrix.data[i]< output->lower_matrix.data[3])
                return 0;
        }
    }
    return 1;
}


int check_not_min_p8(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i!=0 && i!=1){
            if(output->lower_matrix.data[i]< output->upper_matrix.data[0] && output->lower_matrix.data[i]< output->upper_matrix.data[1]){
                return 1;
            }
        }
    }
    return 0;
}


int check_not_min(struct NNet *nnet, struct Interval *output){
	for(int i=0;i<nnet->outputSize;i++){
		if(output->lower_matrix.data[i]<0 && i != nnet->target){
			return 1;
		}
	}
	return 0;
}


int check_not_min_p11(struct NNet *nnet, struct Interval *output){

    if(output->lower_matrix.data[0]<0)
        return 1;

    return 0;
}


int check_min1_p7(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i != 3 && i != 4){
            if(output->data[i]<output->data[3] && output->data[i]<output->data[4])
                return 0;
        }
    }
    return 1;
}


int check_max_constant1(struct NNet *nnet, struct Matrix *output){
    if(output->data[nnet->target]<0.5011){
        return 0;
    }
    return 1;
}


int check_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_not_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}


int check_not_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}


int check_not_min1_p8(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i != 0 && i!=1){
            if(output->data[i]<output->data[0] && output->data[i]<output->data[1])
                return 1;
        }
    }
    return 0;
}


int check_not_min1_p11(struct NNet *nnet, struct Matrix *output){

    if(output->data[0]<0){
        return 1;
    }
    return 0;
}

int check_dave(struct NNet *nnet, struct Interval *output){
    if(output->upper_matrix.data[0]>tan(atan(nnet->out[0])+DAVE_PROPERTY)){
        return 1;
    }
    if(output->upper_matrix.data[0]<tan(atan(nnet->out[0])-DAVE_PROPERTY)){
        return 1;
    }
    else{
        return 0;
    }
}


int check_dave1(struct NNet *nnet, struct Matrix *output){
    if(NEED_PRINT) printf("check between: %f %f\n",tan(atan(nnet->out[0])-DAVE_PROPERTY), tan(atan(nnet->out[0])+DAVE_PROPERTY));
    if(output->data[0]>tan(atan(nnet->out[0])+DAVE_PROPERTY)){
        return 1;
    }
    if(output->data[0]<tan(atan(nnet->out[0])-DAVE_PROPERTY)){
        return 1;
    }
    else{
        return 0;
    }
}


int check_functions(struct NNet *nnet, struct Interval *output, int normalize_output){
    if(PROPERTY>=500){
        return check_dave(nnet, output);
    }
    return check_not_max(nnet, output, normalize_output);
}


int check_functions1(struct NNet *nnet, struct Matrix *output){
    if(PROPERTY>=500){
        return check_dave1(nnet, output);
    }
    return check_not_max1(nnet, output);
}


int tighten_still_overlap(struct NNet *nnet, struct Interval *input, float smear_sum){
    float out1[nnet->outputSize];
    struct Matrix output1 = {out1, nnet->outputSize, 1};
    float out2[nnet->outputSize];
    struct Matrix output2 = {out2, nnet->outputSize, 1};
    forward_prop(nnet, &input->lower_matrix, &output1);
    forward_prop(nnet, &input->upper_matrix, &output2);
    struct Interval output_interval ={output1, output2};
    float upper = 0;
    float lower = 0;
    for(int i=0;i<nnet->outputSize;i++){
        if(i!=nnet->target){
            if(output1.data[i]>output2.data[i]){
                lower = output1.data[i]-smear_sum;
                upper = output2.data[i]+smear_sum;
            }
            else{
                lower = output2.data[i]-smear_sum;
                upper = output1.data[i]+smear_sum;
            }
            
            output1.data[i] = lower;
            output2.data[i] = upper;
        }
    }
    return check_functions(nnet, &output_interval, 1);
}


void *direct_run_check_lp_thread(void *args){
    struct direct_run_check_lp_args *actual_args = args;
    direct_run_check_lp(actual_args->nnet, actual_args->input,\
                     actual_args->grad, actual_args->output_map,
                     actual_args->equation_upper,\
                     actual_args->equation_lower,\
                     actual_args->new_equation_upper,\
                     actual_args->new_equation_lower,\
                     actual_args->wrong_nodes, actual_args->wrong_node_length,\
                     actual_args->sigs,\
                     actual_args->wrong_up_s_up,actual_args->wrong_up_s_low,\
                     actual_args->wrong_low_s_up, actual_args->wrong_low_s_low,\
                     actual_args->target, actual_args->sig,\
                     actual_args->lp, actual_args->rule_num, actual_args->depth);
    return NULL;
}


void *direct_run_check_conv_lp_thread(void *args){
    struct direct_run_check_conv_lp_args *actual_args = args;
    direct_run_check_conv_lp(actual_args->nnet, actual_args->input,\
                     actual_args->output_map,
                     actual_args->equation,\
                     actual_args->equation_err,\
                     actual_args->new_equation,\
                     actual_args->new_equation_err,\
                     actual_args->wrong_nodes, actual_args->wrong_node_length,\
                     actual_args->sigs,\
                     actual_args->equation_conv,\
                     actual_args->equation_conv_err,\
                     actual_args->err_row_conv,\
                     actual_args->target, actual_args->sig,\
                     actual_args->lp, actual_args->rule_num, actual_args->depth);
    return NULL;
}


void *direct_run_check_input_conv_thread(void *args){
    struct direct_run_check_input_conv_args *actual_args = args;
    direct_run_check_input_conv(actual_args->nnet,\
                     actual_args->input,\
                     actual_args->equation,\
                     actual_args->equation_err,\
                     actual_args->new_equation,\
                     actual_args->new_equation_err,\
                     actual_args->depth);
    return NULL;
}


int check_l1(float *input, int inputSize){
    float abs = 0;
    for(int i=0;i<inputSize;i++){
        if(input[i]>=0){
            abs+=input[i];
        }
        else{
            abs-=input[i];
        }
    }
    if(abs<=(float)INF/255.0){
        return 1;
    }
    else {
        return 0;
    }
}


void check_adv(struct NNet* nnet, struct Interval *input){
    float a[nnet->inputSize];
    struct Matrix adv = {a, 1, nnet->inputSize};
    for(int i=0;i<nnet->inputSize;i++){
        float upper = input->upper_matrix.data[i];
        float lower = input->lower_matrix.data[i];
        float middle = (lower+upper)/2;
        a[i] = middle;
    }
    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop(nnet, &adv, &output);
    int is_adv = 0;
    is_adv = check_functions1(nnet, &output);
    if(is_adv && (PROPERTY==504|| PROPERTY==54)){
        if(!check_l1(a, nnet->inputSize)){
            is_adv=0;
        }
    }
    //printMatrix(&adv);
    //printMatrix(&output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(&adv);
        printMatrix(&output);
        int adv_output = 0;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>0 && i != nnet->target){
                    adv_output = i;
            }
        }
        printf("%d ---> %d", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
}


void check_adv1(struct NNet* nnet, struct Matrix *adv){
    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop_conv(nnet, adv, &output);
    //printMatrix(&output);
    int is_adv = 0;
    is_adv = check_functions1(nnet, &output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(&adv);
        printMatrix(&output);
        int adv_output = 0;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>0 && i != nnet->target){
                    adv_output = i;
            }
        }
        printf("%d ---> %d\n", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
}



int pop_queue(int *wrong_nodes, int *wrong_node_length){
    if(*wrong_node_length==0){
        printf("underflow\n");
        can_t_prove = 1;
        return -1;
    }
    int node = wrong_nodes[0];
    for(int i=0;i<*wrong_node_length;i++){
        wrong_nodes[i] = wrong_nodes[i+1];
    }
    *wrong_node_length -= 1;
    return node;
}


int search_queue(int *wrong_nodes, int *wrong_node_length, int node_cnt){
    int wrong_ind=-1;
    for(int wn=0;wn<*wrong_node_length;wn++){
        if(node_cnt == wrong_nodes[wn]){
            wrong_ind = wn;
        }
    }
    return wrong_ind;
}


float max(float a, float b){
    return (a>b)?a:b;
}


float min(float a, float b){
    return (a<b)?a:b;
}


int forward_prop_interval_equation_lp(struct NNet *nnet, struct Interval *input,
                                 struct Interval *grad, int *output_map,
                                 float *equation_upper, float *equation_lower,
                                 float *new_equation_upper, float *new_equation_lower,
                                 int *wrong_nodes, int *wrong_node_length, int *sigs,
                                 float *wrong_up_s_up, float *wrong_up_s_low,
                                 float *wrong_low_s_up, float *wrong_low_s_low,
                                 int target, int sig,
                                 lprec *lp, int *rule_num)
{
    int i,j,k,layer;
    int node_cnt=0;
    int need_to_split = 0;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer

    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    struct Interval equation_inteval = {
            (struct Matrix){(float*)equation_lower, inputSize+1, inputSize},
            (struct Matrix){(float*)equation_upper, inputSize+1, inputSize}
    };
    struct Interval new_equation_inteval = {
            (struct Matrix){(float*)new_equation_lower, inputSize+1, maxLayerSize},
            (struct Matrix){(float*)new_equation_upper, inputSize+1, maxLayerSize}
    };                                       

    float tempVal_upper=0.0, tempVal_lower=0.0;
    float upper_s_lower=0.0, lower_s_upper=0.0;

    float tempVal_upper1=0.0, tempVal_lower1=0.0;
    float upper_s_lower1=0.0, lower_s_upper1=0.0;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }
    float time_spent;

    for (layer = 0; layer<(numLayers); layer++)
    {
        
        memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        
        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];
        float p[weights.col*weights.row];
        float n[weights.col*weights.row];
        memset(p, 0, sizeof(float)*weights.col*weights.row);
        memset(n, 0, sizeof(float)*weights.col*weights.row);
        struct Matrix pos_weights = {p, weights.row, weights.col};
        struct Matrix neg_weights = {n, weights.row, weights.col};
        for(i=0;i<weights.row*weights.col;i++){
            if(weights.data[i]>=0){
                p[i] = weights.data[i];
            }
            else{
                n[i] = weights.data[i];
            }
        }

        matmul(&equation_inteval.upper_matrix, &pos_weights, &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &neg_weights, &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &pos_weights, &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix, &neg_weights, &new_equation_inteval.lower_matrix);
        
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {

float tempVal_upper2 = 0;
float tempVal_lower2 = 0.0;
float lower_s_upper2 =0;
float upper_s_lower2 = 0.0;

for(k=0;k<inputSize;k++){
    if(layer==0){
        if(new_equation_lower[k+i*(inputSize+1)]!=new_equation_upper[k+i*(inputSize+1)]){
            printf("wrong!\n");
        }
    }
    if(new_equation_lower[k+i*(inputSize+1)]>=0){
        tempVal_lower2 += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
        lower_s_upper2 += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
    }
    else{
        tempVal_lower2 += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
        lower_s_upper2 += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
    }
    if(new_equation_upper[k+i*(inputSize+1)]>=0){
        tempVal_upper2 += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
        upper_s_lower2 += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
    }
    else{
        tempVal_upper2 += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
        upper_s_lower2 += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
    }  
}

            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];

tempVal_lower2 += new_equation_lower[inputSize+i*(inputSize+1)];
lower_s_upper2 += new_equation_lower[inputSize+i*(inputSize+1)];
tempVal_upper2 += new_equation_upper[inputSize+i*(inputSize+1)];
upper_s_lower2 += new_equation_upper[inputSize+i*(inputSize+1)];


            int wrong_ind = search_queue(wrong_nodes, wrong_node_length, node_cnt);

            if(node_cnt==target){
                if(sig==1){
                    set_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, sig, inputSize);
                }
                else{
                    set_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, sig, inputSize);
                    for(k=0;k<inputSize+1;k++){
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                    }
                }
                node_cnt ++;
                continue;
            }

            if(layer<(numLayers-1)){
                if(sigs[node_cnt] == 0 && node_cnt != target){
                    //printf("sigs0:%d\n", node_cnt);
                    for(k=0;k<inputSize+1;k++){
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                    }
                    node_cnt++;
                    continue;
                }

                if(sigs[node_cnt] == 1 && node_cnt != target){
                    //printf("sigs1:%d\n", node_cnt);
                    node_cnt++;
                    continue;
                }

                tempVal_upper = wrong_up_s_up[node_cnt];
                lower_s_upper = wrong_low_s_up[node_cnt];
                tempVal_lower = wrong_low_s_low[node_cnt];
                upper_s_lower = wrong_up_s_low[node_cnt];

tempVal_lower2 = max(tempVal_lower2, tempVal_lower);
tempVal_upper2 = min(tempVal_upper2, tempVal_upper);
upper_s_lower2 = max(upper_s_lower, upper_s_lower2);
lower_s_upper2 = min(lower_s_upper, lower_s_upper2);

                //printf("%d %d %d:%f %f %f %f\n", layer, i,node_cnt, tempVal_upper,upper_s_lower, lower_s_upper, tempVal_lower );
                if(wrong_ind!=-1){
                    /*nodes for update the equation*/
                    if(ACCURATE_MODE){
                        if(NEED_PRINT){
                            printf("%d %d %d:%f %f %f %f\n", layer, i, node_cnt, tempVal_upper, upper_s_lower, lower_s_upper, tempVal_lower );
                        }
                        if(set_wrong_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, inputSize, MAX, &tempVal_upper1)){
                            if(NEED_PRINT) printf("%d %d, uu, wrong\n",layer,i );
                            return 0;
                        }
                        if(set_wrong_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, inputSize, MIN, &upper_s_lower1)){
                            if(NEED_PRINT) printf("%d %d, ul, wrong\n",layer,i );
                            return 0;
                        }
                        if(set_wrong_node_constraints(lp, new_equation_lower, i*(inputSize+1), rule_num, inputSize, MAX, &lower_s_upper1)){
                            if(NEED_PRINT) printf("%d %d, lu, wrong\n",layer,i );
                            return 0;
                        }
                        if(set_wrong_node_constraints(lp, new_equation_lower, i*(inputSize+1), rule_num, inputSize, MIN, &tempVal_lower1)){
                            if(NEED_PRINT) printf("%d %d, ll, wrong\n",layer,i );
                            return 0;
                        }

                        if (tempVal_upper1<=0.0){
                            for(k=0;k<inputSize+1;k++){
                                new_equation_upper[k+i*(inputSize+1)] = 0;
                                new_equation_lower[k+i*(inputSize+1)] = 0;
                            }
                            R[layer][i] = 0;
                        }
                        else if(tempVal_lower1>=0.0){
                            R[layer][i] = 2;
                        }
                        else{
                            //printf("wrong node: ");
                            if(upper_s_lower1<0.0){
                                for(k=0;k<inputSize+1;k++){
                                    new_equation_upper[k+i*(inputSize+1)] =\
                                                            new_equation_upper[k+i*(inputSize+1)]*\
                                                            tempVal_upper1 / (tempVal_upper1-upper_s_lower1);
                                }
                                new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper1*upper_s_lower1/\
                                                                    (tempVal_upper1-upper_s_lower1);
                            }

                            if(lower_s_upper1<0.0){
                                for(k=0;k<inputSize+1;k++){
                                    new_equation_lower[k+i*(inputSize+1)] = 0;
                                }
                            }
                            else{
                                for(k=0;k<inputSize+1;k++){
                                    new_equation_lower[k+i*(inputSize+1)] =\
                                                            new_equation_lower[k+i*(inputSize+1)]*\
                                                            lower_s_upper1 / (lower_s_upper1- tempVal_lower1);
                                }
                            }
                            R[layer][i] = 1;
                        }
                        if(NEED_PRINT){
                            //printf("original: %d %d %d:%f %f %f %f\n", layer, i,node_cnt, tempVal_upper,upper_s_lower, lower_s_upper, tempVal_lower);
                            //printf("no cons: %d %d %d:%f %f %f %f\n", layer, i,node_cnt, tempVal_upper2,upper_s_lower2, lower_s_upper2, tempVal_lower2);
                            printf("%d %d %d:%f %f %f %f\n", layer, i, node_cnt, tempVal_upper1, upper_s_lower1, lower_s_upper1, tempVal_lower1 );
                        }
                    }
                    else{
                        for(k=0;k<inputSize+1;k++){
                            new_equation_upper[k+i*(inputSize+1)] =\
                                                    new_equation_upper[k+i*(inputSize+1)]*\
                                                    tempVal_upper2 / (tempVal_upper2-tempVal_lower2);
                        }
                        new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper2*tempVal_lower2/\
                                                            (tempVal_upper2-tempVal_lower2);

                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] =\
                                                    new_equation_lower[k+i*(inputSize+1)]*\
                                                    tempVal_upper2 / (tempVal_upper2- tempVal_lower2);
                        }
                    }
                }
                else{
                    if(tempVal_upper<0){
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                            new_equation_upper[k+i*(inputSize+1)] = 0;
                        }
                    }
                }
            }
            else{
                if(i!=nnet->target){
                    //gettimeofday(&start, NULL);
                    float upper = 0.0;
                    float input_prev[inputSize];
                    struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                    memset(input_prev, 0, sizeof(float)*inputSize);
                    float o[outputSize];
                    struct Matrix output_matrix = {o, outputSize, 1};
                    memset(o, 0, sizeof(float)*outputSize);
                    if(output_map[i]){
                        if(!set_output_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, inputSize, MAX, &upper, input_prev)){
                            need_to_split = 1;
                            output_map[i] = 1;
                            if(NEED_PRINT){
                                printf("%d--Objective value: %f\n", i, upper);
                            }
                            check_adv1(nnet, &input_prev_matrix);
                            if(adv_found){
                                return 0;
                            }
                        }
                        else{
                            output_map[i] = 0;
                            if(NEED_PRINT){
                                printf("%d--unsat\n", i);
                            }
                        }
                    }/*
                    gettimeofday(&finish, NULL);
                    time_spent = ((float)(finish.tv_sec-start.tv_sec)*1000000 +\
                        (float)(finish.tv_usec-start.tv_usec)) / 1000000;
                    printf("%d: %f \n",i, time_spent);
                    */
                }
            }
            node_cnt ++;
        }
        
        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
        
    }

    return need_to_split;
}



int forward_prop_interval_equation_conv_lp(struct NNet *nnet,\
                    struct Interval *input, int *output_map,
                    float *equation, float *equation_err,
                    float *new_equation, float *new_equation_err, int *sigs,
                    float *equation_conv, float *equation_conv_err,\
                    float err_row_conv, int target, int sig,
                    lprec *lp, int *rule_num)
{
    int i,j,k,layer;
    int node_cnt=0;
    int need_to_split=0;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = 0;

    for(layer=0;layer<nnet->numLayers;layer++){
        if(nnet->layerTypes[layer]==0){
            if(nnet->layerSizes[layer]>maxLayerSize){
                maxLayerSize = nnet->layerSizes[layer];
            }
        }
    }

    // equation is the temp equation for each layer

    memset(equation,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_err,0,sizeof(float)*ERR_NODE*maxLayerSize);

    struct Matrix equation_matrix = {(float*)equation, inputSize+1, inputSize};
    struct Matrix new_equation_matrix = {(float*)new_equation, inputSize+1, inputSize};

    //The actual row number for index is ERR_NODE, but the row used for matmul is the true current error node err_row
    // This is because all the other lines are 0
    struct Matrix equation_err_matrix = {(float*)equation_err, ERR_NODE, inputSize};
    struct Matrix new_equation_err_matrix = {(float*)new_equation_err, ERR_NODE, inputSize};  

    float tempVal_upper=0.0, tempVal_lower=0.0;
    float upper_s_lower=0.0, lower_s_upper=0.0; 

    memset(equation_err,0, sizeof(float)*ERR_NODE*maxLayerSize);

    //err_row is the number that is wrong before current layer
    int err_row=0;
    int wrong_node_length = 0;
    for (layer = 0; layer<(numLayers); layer++)
    {
        
        memset(new_equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_err,0,sizeof(float)*ERR_NODE*maxLayerSize);
        
        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];

        if(nnet->layerTypes[layer]==0 && nnet->layerTypes[layer-1]==1){
            memcpy(new_equation, equation_conv, sizeof(float)*(inputSize+1)*maxLayerSize);
            memcpy(new_equation_err, equation_conv_err, sizeof(float)*ERR_NODE*maxLayerSize);
            err_row = err_row_conv;
            wrong_node_length = err_row;
            equation_matrix.col = new_equation_matrix.col = nnet->layerSizes[layer+1];
        }
        else if(nnet->layerTypes[layer]==1){
            node_cnt += nnet->layerSizes[layer+1];
            continue;
        }
        else{
            matmul(&equation_matrix, &weights, &new_equation_matrix);
            if(err_row>0){
                equation_err_matrix.row = ERR_NODE;
                matmul(&equation_err_matrix, &weights, &new_equation_err_matrix);
            }
        }

        new_equation_err_matrix.row = equation_err_matrix.row = err_row;
        equation_err_matrix.col = new_equation_err_matrix.col = nnet->layerSizes[layer+1];

        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            tempVal_upper = tempVal_lower = 0.0;

            if(NEED_OUTWARD_ROUND){
                for(k=0;k<inputSize;k++){
                    if(new_equation[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                        tempVal_upper += new_equation[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                    }
                    else{
                        tempVal_lower += new_equation[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                        tempVal_upper += new_equation[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                    } 
                }
            }
            else{
                for(k=0;k<inputSize;k++){
                    //if(i==2 && layer==3) printf("%f\n",new_equation[k+i*(inputSize+1)]);
                    if(new_equation[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                        tempVal_upper += new_equation[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }
                    else{
                        tempVal_lower += new_equation[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                        tempVal_upper += new_equation[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    } 
                }
            }
            
            new_equation[inputSize+i*(inputSize+1)] += bias.data[i];
            tempVal_lower += new_equation[inputSize+i*(inputSize+1)];
            tempVal_upper += new_equation[inputSize+i*(inputSize+1)];

            if(err_row>0){

                for(int err_ind=0;err_ind<err_row;err_ind++){
                    //if(i==2 && layer==3 && err_ind<400) printf("%f\n",new_equation_err[err_ind+i*ERR_NODE]);
                    if(new_equation_err[err_ind+i*ERR_NODE]>0){
                        tempVal_upper += new_equation_err[err_ind+i*ERR_NODE];
                    }
                    else{
                        tempVal_lower += new_equation_err[err_ind+i*ERR_NODE];
                    }
                }
            }

            if(node_cnt==target){
                if(sig==1){
                    set_node_constraints(lp, new_equation, i*(inputSize+1), rule_num, sig, inputSize);
                }
                else{
                    set_node_constraints(lp, new_equation, i*(inputSize+1), rule_num, sig, inputSize);
                    for(k=0;k<inputSize+1;k++){
                        new_equation[k+i*(inputSize+1)] = 0;
                    }
                    if(err_row>0){
                        for(int err_ind=0;err_ind<err_row;err_ind++){
                            new_equation_err[err_ind+i*ERR_NODE] = 0;
                        }
                    }
                }
                node_cnt ++;
                continue;
            }

            //Perform ReLU
            //if(layer==3) printf("%d %d %d %f %f\n", layer, i, node_cnt, tempVal_lower, tempVal_upper);
            if(layer<(numLayers-1)){
                if(sigs[node_cnt] == 0 && node_cnt != target){
                    //printf("sigs0:%d\n", node_cnt);
                    for(k=0;k<inputSize+1;k++){
                        for(k=0;k<inputSize+1;k++){
                            new_equation[k+i*(inputSize+1)] = 0;
                        }
                        if(err_row>0){
                            for(int err_ind=0;err_ind<err_row;err_ind++){
                                new_equation_err[err_ind+i*ERR_NODE] = 0;
                            }
                        }
                    }
                    node_cnt++;
                    continue;
                }

                if(sigs[node_cnt] == 1 && node_cnt != target){
                    //printf("sigs1:%d\n", node_cnt);
                    node_cnt++;
                    continue;
                }

                if (tempVal_upper<=0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation[k+i*(inputSize+1)] = 0;
                    }
                    if(err_row>0){
                        for(int err_ind=0;err_ind<err_row;err_ind++){
                            new_equation_err[err_ind+i*ERR_NODE] = 0;
                        }
                    }
                }
                else if(tempVal_lower>=0.0){

                }
                else{
                    wrong_node_length += 1;
                    //printf("wrong: %d,%d:%f, %f\n",layer, i, tempVal_lower, tempVal_upper);
                    
                    //printf("wrong node: ");
                    for(k=0;k<inputSize+1;k++){
                        new_equation[k+i*(inputSize+1)] =\
                                                new_equation[k+i*(inputSize+1)]*\
                                                tempVal_upper / (tempVal_upper - tempVal_lower);
                    }
                    if(err_row>0){
                        //printf("err_row:%d ul: %f\n",err_row,  tempVal_upper / (tempVal_upper - tempVal_lower));
                        for(int err_ind=0;err_ind<err_row;err_ind++){
                            new_equation_err[err_ind+i*ERR_NODE] *= tempVal_upper / (tempVal_upper - tempVal_lower);
                        }
                    }
                    
                    new_equation_err[wrong_node_length-1+i*ERR_NODE] -= tempVal_upper*tempVal_lower/\
                                                        (tempVal_upper-tempVal_lower);
                    //new_equation_err[new_equation_err_matrix.row*nnet->layerSizes[layer+1]+i] -= tempVal_upper*tempVal_lower/\
                                                        (tempVal_upper-tempVal_lower);
                }
            }
            else{
                if(PROPERTY<500){
                    //mnist checking function is different from self-driving car
                    if(i!=nnet->target){
                        float upper_err=0, lower_err=0;
                        // original output - target output
                        for(int k=0;k<inputSize+1;k++){
                            new_equation[k+i*(inputSize+1)] -= new_equation[k+nnet->target*(inputSize+1)]; 
                        }
                        
                        for(int err_ind=0;err_ind<err_row;err_ind++){
                            new_equation_err[err_ind+i*ERR_NODE] -= new_equation_err[err_ind+nnet->target*ERR_NODE];
                            if(new_equation_err[err_ind+i*ERR_NODE]>0){
                                upper_err += new_equation_err[err_ind+i*ERR_NODE];
                            }
                            else{
                                lower_err += new_equation_err[err_ind+i*ERR_NODE];
                            }
                        }
                        new_equation[inputSize+i*(inputSize+1)] += upper_err;
                        //printf("%d %f, %f, %f,%f\n",i,upper_err, tempVal_lower, tempVal_upper, new_equation[inputSize+i*(inputSize+1)]);
                        

                        //gettimeofday(&start, NULL);
                        float upper = 0.0;
                        float input_prev[inputSize];
                        struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                        memset(input_prev, 0, sizeof(float)*inputSize);
                        float o[outputSize];
                        struct Matrix output_matrix = {o, outputSize, 1};
                        memset(o, 0, sizeof(float)*outputSize);
                        if(output_map[i]){
                            if(!set_output_constraints(lp, new_equation, i*(inputSize+1), rule_num, inputSize, MAX, &upper, input_prev)){
                                need_to_split = 1;
                                output_map[i] = 1;
                                if(NEED_PRINT){
                                    printf("%d--Objective value: %f\n", i, upper);
                                }
                                check_adv1(nnet, &input_prev_matrix);
                                if(adv_found){
                                    return 0;
                                }
                            }
                            
                            if(!need_to_split){
                                output_map[i] = 0;
                                if(NEED_PRINT){
                                    printf("%d--unsat\n", i);
                                }
                            }
                        }

                    }
                }
                else{
                    float upper_err=0, lower_err=0;
                    for(int err_ind=0;err_ind<err_row;err_ind++){
                        //new_equation_err[err_ind+i*ERR_NODE] -= new_equation_err[err_ind+nnet->target*ERR_NODE];
                        if(new_equation_err[err_ind+i*ERR_NODE]>0){
                            upper_err += new_equation_err[err_ind+i*ERR_NODE];
                        }
                        else{
                            lower_err += new_equation_err[err_ind+i*ERR_NODE];
                        }
                    }
                    new_equation[inputSize+i*(inputSize+1)] += upper_err;
                    if(NEED_PRINT) printf("%d %f, tempVal_lower:%f, tempVal_upper:%f,%f\n",i,upper_err, tempVal_lower, tempVal_upper, new_equation[inputSize+i*(inputSize+1)]);
                    

                    //gettimeofday(&start, NULL);
                    float upper = 0.0;
                    float input_prev[inputSize];
                    struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                    memset(input_prev, 0, sizeof(float)*inputSize);
                    float o[outputSize];
                    struct Matrix output_matrix = {o, outputSize, 1};
                    memset(o, 0, sizeof(float)*outputSize);
                    if(output_map[i]){
                        new_equation[inputSize+i*(inputSize+1)] -= tan(atan(nnet->out[0])+DAVE_PROPERTY);
                        if(!set_output_constraints(lp, new_equation, i*(inputSize+1), rule_num, inputSize, MAX, &upper, input_prev)){
                            need_to_split = 1;
                            output_map[i] = 1;
                            if(NEED_PRINT){
                                printf("%d--Objective upper value: %f\n", i, upper);
                            }
                            check_adv1(nnet, &input_prev_matrix);
                            if(adv_found){
                                return 0;
                            }
                        }
                        new_equation[inputSize+i*(inputSize+1)] += tan(atan(nnet->out[0])+DAVE_PROPERTY);
                        new_equation[inputSize+i*(inputSize+1)] -= upper_err;
                        new_equation[inputSize+i*(inputSize+1)] += lower_err;
                        new_equation[inputSize+i*(inputSize+1)] -= tan(atan(nnet->out[0])-DAVE_PROPERTY);
                        if(!set_output_constraints(lp, new_equation, i*(inputSize+1), rule_num, inputSize, MIN, &upper, input_prev)){
                            need_to_split = 1;
                            output_map[i] = 1;
                            if(NEED_PRINT){
                                printf("%d--Objective lower value: %f\n", i, upper);
                            }
                            check_adv1(nnet, &input_prev_matrix);
                            if(adv_found){
                                return 0;
                            }
                        }
                        
                        if(!need_to_split){
                            output_map[i] = 0;
                            if(NEED_PRINT){
                                printf("%d--unsat\n", i);
                            }
                        }
                    }

                    
                }
                
                
            }
            node_cnt++;
        }
    
        //printf("\n");
        memcpy(equation, new_equation, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_err, new_equation_err, sizeof(float)*(ERR_NODE)*maxLayerSize);
        equation_matrix.row = new_equation_matrix.row;
        equation_matrix.col = new_equation_matrix.col;
        equation_err_matrix.row = new_equation_err_matrix.row;
        equation_err_matrix.col = new_equation_err_matrix.col;
        err_row = wrong_node_length;
    }

    return need_to_split;
}


int direct_run_check_lp(struct NNet *nnet, struct Interval *input,
                     struct Interval *grad, int *output_map,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *wrong_up_s_up, float *wrong_up_s_low,
                     float *wrong_low_s_up, float *wrong_low_s_low,
                     int target, int sig,
                     lprec *lp, int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return 0;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    if(depth<=3){
        solve(lp);
    }

    int isOverlap = 0;

    isOverlap = forward_prop_interval_equation_lp(nnet, input,\
                                 grad, output_map,\
                                 equation_upper, equation_lower,\
                                 new_equation_upper, new_equation_lower,\
                                 wrong_nodes, wrong_node_length, sigs,\
                                 wrong_up_s_up, wrong_up_s_low,\
                                 wrong_low_s_up, wrong_low_s_low,\
                                 target, sig,\
                                 lp, rule_num);

    if(depth<=PROGRESS_DEPTH && !isOverlap){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        pthread_mutex_unlock(&lock);
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            if(p>depth){
                total_progress[p-1] -= pow(2,(p-depth));
            }
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
    }

    if(isOverlap && !NEED_FOR_ONE_RUN){
        if(NEED_PRINT) printf("depth:%d, sig:%d Need to split!\n\n", depth, sig);
        isOverlap = split_interval_lp(nnet, input,\
                         grad, output_map,
                         equation_upper, equation_lower,\
                         new_equation_upper, new_equation_lower,\
                         wrong_nodes, wrong_node_length, sigs,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         lp, rule_num, depth);
    }
    else{
        if(!adv_found)
            if(NEED_PRINT) printf("depth:%d, sig:%d, UNSAT, great!\n\n", depth, sig);
            pthread_mutex_lock(&lock);
                avg_depth -= (avg_depth) / AVG_WINDOW;
                avg_depth += depth / AVG_WINDOW;
            pthread_mutex_unlock(&lock);
    }
    return isOverlap;
}


int direct_run_check_conv_lp(struct NNet *nnet, struct Interval *input, int *output_map,
                     float *equation, float *equation_err,
                     float *new_equation, float *new_equation_err,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *equation_conv, float *equation_conv_err, float err_row_conv,
                     int target, int sig,
                     lprec *lp, int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return 0;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    if(depth<=3){
        solve(lp);
    }

    int isOverlap = 0;

    isOverlap = forward_prop_interval_equation_conv_lp(nnet, input, output_map,\
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 sigs, equation_conv, equation_conv_err, err_row_conv,\
                                 target, sig,\
                                 lp, rule_num);


    if(depth<=PROGRESS_DEPTH && !isOverlap){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        pthread_mutex_unlock(&lock);
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            if(p>depth){
                total_progress[p-1] -= pow(2,(p-depth));
            }
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
    }

    if(isOverlap && !NEED_FOR_ONE_RUN){
        if(NEED_PRINT) printf("depth:%d, sig:%d Need to split!\n\n", depth, sig);
        isOverlap = split_interval_conv_lp(nnet, input, output_map,
                         equation, equation_err,\
                         new_equation, new_equation_err,\
                         wrong_nodes, wrong_node_length, sigs,\
                         equation_conv, equation_conv_err, err_row_conv,\
                         lp, rule_num, depth);
    }
    else{
        if(!adv_found)
            if(NEED_PRINT) printf("depth:%d, sig:%d, UNSAT, great!\n\n",\
                        depth, sig);
            pthread_mutex_lock(&lock);
                avg_depth -= (avg_depth) / AVG_WINDOW;
                avg_depth += depth / AVG_WINDOW;
            pthread_mutex_unlock(&lock);
    }
    return isOverlap;
}

int split_interval_lp(struct NNet *nnet, struct Interval *input,
                     struct Interval *grad, int *output_map,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *wrong_up_s_up, float *wrong_up_s_low,
                     float *wrong_low_s_up, float *wrong_low_s_low,
                     lprec *lp, int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    
    if(depth>=22){
        can_t_prove = 1;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    if(depth==0){
        memset(progress_list, 0, PROGRESS_DEPTH*sizeof(int));
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            total_progress[p-1] = pow(2, p);
        }
    }

    depth ++;

    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;
    int outputSize = nnet->outputSize;

    int target = 0;
    target = pop_queue(wrong_nodes, wrong_node_length);
    int sig = 0;
    int isOverlap1, isOverlap2;

    float *equation_upper1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *equation_lower1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *new_equation_upper1 = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
    float *new_equation_lower1 = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);
    lprec *lp1, *lp2;
    //write_lp(lp, "model.lp");
    //lp1 = read_LP("model.lp", IMPORTANT, NULL);
    //lp2 = read_LP("model.lp", IMPORTANT, NULL);
    lp1 = copy_lp(lp);
    lp2 = copy_lp(lp);

    int rule_num1 = *rule_num;
    int rule_num2 = *rule_num;

    int wrong_node_length1 = *wrong_node_length;
    int wrong_node_length2 = *wrong_node_length; 
    int wrong_nodes1[wrong_node_length1];
    int wrong_nodes2[wrong_node_length2];
    memcpy(wrong_nodes1, wrong_nodes, sizeof(int)*wrong_node_length1);
    memcpy(wrong_nodes2, wrong_nodes, sizeof(int)*wrong_node_length2);

    int output_map1[outputSize];
    int output_map2[outputSize];
    memcpy(output_map1, output_map, sizeof(int)*outputSize);
    memcpy(output_map2, output_map, sizeof(int)*outputSize);

    int sigSize = 0; 
    for(int layer=1;layer<nnet->numLayers;layer++){
        sigSize += nnet->layerSizes[layer];
    }

    int sigs1[sigSize];
    int sigs2[sigSize];

    memcpy(sigs1, sigs, sizeof(int)*sigSize);
    memcpy(sigs2, sigs, sizeof(int)*sigSize);

    int sig1,sig2;
    sig1 = 1;
    sig2 = 0;
    sigs1[target] = 1;
    sigs2[target] = 0;
    pthread_mutex_lock(&lock);

    if ((depth <= avg_depth- MIN_DEPTH_PER_THREAD) &&\
            (count<=MAX_THREAD) && !NEED_FOR_ONE_RUN) {
        pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;
        struct direct_run_check_lp_args args1 = {nnet, input,\
                                 grad, output_map1,
                                 equation_upper, equation_lower,\
                                 new_equation_upper, new_equation_lower,\
                                 wrong_nodes1, &wrong_node_length1, sigs1,\
                                 wrong_up_s_up, wrong_up_s_low,\
                                 wrong_low_s_up, wrong_low_s_low,\
                                 target, sig1,\
                                 lp1, &rule_num1, depth};

        struct direct_run_check_lp_args args2 = {nnet, input,\
                                 grad, output_map2,
                                 equation_upper1, equation_lower1,\
                                 new_equation_upper1, new_equation_lower1,\
                                 wrong_nodes2, &wrong_node_length2, sigs2,\
                                 wrong_up_s_up, wrong_up_s_low,\
                                 wrong_low_s_up, wrong_low_s_low,\
                                 target, sig2,\
                                 lp2, &rule_num2, depth};

        pthread_create(&workers1, NULL, direct_run_check_lp_thread, &args1);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, NULL, direct_run_check_lp_thread, &args2);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_join(workers1, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld done %d\n",syscall(SYS_gettid), count);
        pthread_join(workers2, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

        isOverlap1 = 0;
        isOverlap2 = 0;

    }
    else {
        pthread_mutex_unlock(&lock);
        isOverlap1 = direct_run_check_lp(nnet, input,\
                         grad, output_map1,
                         equation_upper, equation_lower,\
                         new_equation_upper, new_equation_lower,\
                         wrong_nodes1, &wrong_node_length1, sigs1,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         target, sig1,\
                         lp1, &rule_num1, depth);

        isOverlap2 = direct_run_check_lp(nnet, input,\
                         grad, output_map2,
                         equation_upper1, equation_lower1,\
                         new_equation_upper1, new_equation_lower1,\
                         wrong_nodes2, &wrong_node_length2, sigs2,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         target, sig2,\
                         lp2, &rule_num2, depth);
    }

    free(equation_upper1);
    free(equation_lower1);
    free(new_equation_upper1);
    free(new_equation_lower1);
    delete_lp(lp1);
    delete_lp(lp2);

    int result = isOverlap1 || isOverlap2;
    depth --;

    if(!result && depth<=PROGRESS_DEPTH){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
        pthread_mutex_unlock(&lock);
    }

    return result;
}


int split_interval_conv_lp(struct NNet *nnet, struct Interval *input,\
                    int *output_map, float *equation, float *equation_err,\
                    float *new_equation, float *new_equation_err,
                    int *wrong_nodes, int *wrong_node_length, int *sigs,
                    float *equation_conv, float *equation_conv_err,\
                    float err_row_conv, lprec *lp, int *rule_num, int depth)
{
    int layer;

    pthread_mutex_lock(&lock);

    if (adv_found) {
        pthread_mutex_unlock(&lock);
        return 0;
    }
    
    if (depth >= 18) {
        printf("depth is limited by %d\n", 22);
        can_t_prove = 1;
    }

    if (can_t_prove) {
        pthread_mutex_unlock(&lock);
        return 0;
    }

    pthread_mutex_unlock(&lock);

    if (depth == 0) {
        memset(progress_list, 0, PROGRESS_DEPTH*sizeof(int));
        
        for (int p=1;p<PROGRESS_DEPTH+1;p++) {
            total_progress[p-1] = pow(2, p);
        }

    }

    depth ++;

    int inputSize = nnet->inputSize;
    int maxLayerSize = 0;
    int outputSize = nnet->outputSize;

    for (layer=0;layer<nnet->numLayers;layer++) {

        if (nnet->layerTypes[layer] == 0) {

            if(nnet->layerSizes[layer]>maxLayerSize){
                maxLayerSize = nnet->layerSizes[layer];
            }

        }

    }

    int target = 0;
    target = pop_queue(wrong_nodes, wrong_node_length);
    int sig = 0;
    int isOverlap1, isOverlap2;

    float *equation1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *equation_err1 = (float*)malloc(sizeof(float) *\
                            ERR_NODE*maxLayerSize);
    float *new_equation1 = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
    float *new_equation_err1 = (float*)malloc(sizeof(float) *\
                                    ERR_NODE*maxLayerSize);

    lprec *lp1, *lp2;
    //write_lp(lp, "model.lp");
    //lp1 = read_LP("model.lp", IMPORTANT, NULL);
    //lp2 = read_LP("model.lp", IMPORTANT, NULL);
    lp1 = copy_lp(lp);
    lp2 = copy_lp(lp);

    int rule_num1 = *rule_num;
    int rule_num2 = *rule_num;

    int wrong_node_length1 = *wrong_node_length;
    int wrong_node_length2 = *wrong_node_length; 
    int wrong_nodes1[wrong_node_length1];
    int wrong_nodes2[wrong_node_length2];

    memcpy(wrong_nodes1, wrong_nodes, sizeof(int)*wrong_node_length1);
    memcpy(wrong_nodes2, wrong_nodes, sizeof(int)*wrong_node_length2);

    int output_map1[outputSize];
    int output_map2[outputSize];
    memcpy(output_map1, output_map, sizeof(int)*outputSize);
    memcpy(output_map2, output_map, sizeof(int)*outputSize);

    int sigSize = 0; 
    for(int layer=1;layer<nnet->numLayers;layer++){
        sigSize += nnet->layerSizes[layer];
    }

    int sigs1[sigSize];
    int sigs2[sigSize];

    memcpy(sigs1, sigs, sizeof(int)*sigSize);
    memcpy(sigs2, sigs, sizeof(int)*sigSize);

    int sig1,sig2;
    sig1 = 1;
    sig2 = 0;
    sigs1[target] = 1;
    sigs2[target] = 0;
    pthread_mutex_lock(&lock);

    if ((depth <= avg_depth- MIN_DEPTH_PER_THREAD) &&\
                (count<=MAX_THREAD) && !NEED_FOR_ONE_RUN) {
        pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;

        struct direct_run_check_conv_lp_args args1 = {nnet, input, output_map1,
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 wrong_nodes1, &wrong_node_length1, sigs1,\
                                 equation_conv, equation_conv_err, err_row_conv,\
                                 target, sig1,\
                                 lp1, &rule_num1, depth};

        struct direct_run_check_conv_lp_args args2 = {nnet, input, output_map2,
                                 equation1, equation_err1,\
                                 new_equation1, new_equation_err1,\
                                 wrong_nodes2, &wrong_node_length2, sigs2,\
                                 equation_conv, equation_conv_err, err_row_conv,\
                                 target, sig2,\
                                 lp2, &rule_num2, depth};

        pthread_create(&workers1, NULL, direct_run_check_conv_lp_thread, &args1);
       
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);

        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, NULL, direct_run_check_conv_lp_thread, &args2);
        
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);

        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_join(workers1, NULL);

        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

        //printf ( "pid1: %ld done %d\n",syscall(SYS_gettid), count);
        pthread_join(workers2, NULL);

        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

        isOverlap1 = 0;
        isOverlap2 = 0;

    }
    else {
        pthread_mutex_unlock(&lock);
        isOverlap1 = direct_run_check_conv_lp(nnet, input, output_map1,
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 wrong_nodes1, &wrong_node_length1, sigs1,\
                                 equation_conv, equation_conv_err, err_row_conv,\
                                 target, sig1,\
                                 lp1, &rule_num1, depth);

        isOverlap2 = direct_run_check_conv_lp(nnet, input, output_map2,
                                 equation1, equation_err1,\
                                 new_equation1, new_equation_err1,\
                                 wrong_nodes2, &wrong_node_length2, sigs2,\
                                 equation_conv, equation_conv_err, err_row_conv,\
                                 target, sig2,\
                                 lp2, &rule_num2, depth);
    }

    free(equation1);
    free(equation_err1);
    free(new_equation1);
    free(new_equation_err1);
    delete_lp(lp1);
    delete_lp(lp2);

    int result = isOverlap1 || isOverlap2;
    depth --;

    if(!result && depth<=PROGRESS_DEPTH){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
        pthread_mutex_unlock(&lock);
    }

    return result;
}



int split_input_interval_conv(struct NNet *nnet, struct Interval *input,
                     float *equation, float *equation_err,
                     float *new_equation, float *new_equation_err,
                     int depth)
{
    int layer;

    pthread_mutex_lock(&lock);

    if (adv_found) {
        pthread_mutex_unlock(&lock);
        return 0;
    }
    
    if (depth >= 15) {
        can_t_prove = 1;
    }

    if (can_t_prove) {
        pthread_mutex_unlock(&lock);
        return 0;
    }

    pthread_mutex_unlock(&lock);

    if (depth == 0) {
        memset(progress_list, 0, PROGRESS_DEPTH*sizeof(int));

        for (int p=1;p<PROGRESS_DEPTH+1;p++) {
            total_progress[p-1] = pow(2, p);
        }

    }

    depth ++;

    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;
    int outputSize = nnet->outputSize;

    int isOverlap1, isOverlap2;
    float middle[nnet->inputSize];

    for (int i=0;i<nnet->inputSize;i++) {
        middle[i] = (input->upper_matrix.data[i]+input->lower_matrix.data[i])/2;
    }

    struct Matrix middle_matrix = {middle, 1, nnet->inputSize};

    check_adv1(nnet, &input->upper_matrix);
    check_adv1(nnet, &input->lower_matrix);

    if (adv_found) {
        return 0;
    }


    struct Interval input1 = {input->lower_matrix, middle_matrix};
    struct Interval input2 = {middle_matrix, input->upper_matrix};

    pthread_mutex_lock(&lock);

    if ((depth <= avg_depth- MIN_DEPTH_PER_THREAD) &&\
            (count<=MAX_THREAD) && !NEED_FOR_ONE_RUN) {
        float *equation1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
        float *equation_err1 = (float*)malloc(sizeof(float) *\
                                ERR_NODE*maxLayerSize);
        float *new_equation1 = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);
        float *new_equation_err1 = (float*)malloc(sizeof(float) *\
                                        ERR_NODE*maxLayerSize);

        pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;
        struct direct_run_check_input_conv_args args1 = {nnet, &input1,\
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 depth};

        struct direct_run_check_input_conv_args args2 = {nnet, &input2,\
                                 equation1, equation_err1,\
                                 new_equation1, new_equation_err1,\
                                 depth};

        pthread_create(&workers1, NULL, direct_run_check_input_conv_thread, &args1);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, NULL, direct_run_check_input_conv_thread, &args2);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_join(workers1, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld done %d\n",syscall(SYS_gettid), count);
        pthread_join(workers2, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

        isOverlap1 = 0;
        isOverlap2 = 0;

        free(equation1);
        free(equation_err1);
        free(new_equation1);
        free(new_equation_err1);
    }
    else {
        pthread_mutex_unlock(&lock);
        isOverlap1 = direct_run_check_input_conv(nnet, &input1,
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 depth);

        isOverlap2 = direct_run_check_input_conv(nnet, &input2,
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 depth);
    }

    int result = isOverlap1 || isOverlap2;
    depth --;

    if (!result && depth<=PROGRESS_DEPTH) {
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        fprintf(stderr, " progress: ");

        for (int p=1;p<PROGRESS_DEPTH+1;p++) {
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }

        fprintf(stderr, "\n");
        pthread_mutex_unlock(&lock);
    }

    return result;
}


int direct_run_check_input_conv(struct NNet *nnet,\
                     struct Interval *input,\
                     float *equation, float *equation_err,\
                     float *new_equation, float *new_equation_err,\
                     int depth)
{
    int outputSize = nnet->outputSize;
    int inputSize = nnet->inputSize;
    pthread_mutex_lock(&lock);

    if (adv_found) {
        pthread_mutex_unlock(&lock);
        return 0;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return 0;
    }

    pthread_mutex_unlock(&lock);

    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {
                (struct Matrix){o_lower, outputSize, 1},
                (struct Matrix){o_upper, outputSize, 1}
            };

    int isOverlap = 0;

    forward_prop_input_interval_equation_linear_conv(nnet,\
                        input, &output_interval,\
                        equation, equation_err,\
                        new_equation, new_equation_err);

    if (NEED_PRINT && outputSize <= 10) {
        printMatrix(&output_interval.upper_matrix);
        printMatrix(&output_interval.lower_matrix);
    }

    isOverlap = check_functions(nnet, &output_interval,1);

    if (depth <= PROGRESS_DEPTH && !isOverlap) {
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        pthread_mutex_unlock(&lock);
        fprintf(stderr, " progress: ");

        for (int p=1;p<PROGRESS_DEPTH+1;p++) {

            if (p>depth) {
                total_progress[p-1] -= pow(2,(p-depth));
            }

            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }

        fprintf(stderr, "\n");
    }

    if (isOverlap && !NEED_FOR_ONE_RUN) {
        
        if(NEED_PRINT) printf("depth:%d, Need to split!\n\n", depth);

        isOverlap = split_input_interval_conv(nnet, input,
                         equation, equation_err,\
                         new_equation, new_equation_err,\
                         depth);
    }
    else {

        if (!adv_found) {
            if(NEED_PRINT) printf("depth:%d,, UNSAT, great!\n\n", depth);
            pthread_mutex_lock(&lock);
                avg_depth -= (avg_depth) / AVG_WINDOW;
                avg_depth += depth / AVG_WINDOW;
            pthread_mutex_unlock(&lock);
        }

    }

    return isOverlap;
}
