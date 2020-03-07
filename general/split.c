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
bool adv_found = false;
bool can_t_prove = false;
int count = 0;
int thread_tot_cnt  = 0;
int smear_cnt = 0;

int progress = 0;
int MAX_DEPTH = 30;

float avg_depth = 50;

int progress_list[PROGRESS_DEPTH];
int total_progress[PROGRESS_DEPTH];


 /*
  * You need to customize your own checking function.
  * Here is a couple of sample functions that you could use.
  */
bool check_not_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]>0 && i != nnet->target){
            return true;
        }
    }
    return false;
}


bool check_max_constant(struct NNet *nnet, struct Interval *output){
    if(output->upper_matrix.data[nnet->target]>0.5011){
        return true;
    }
    else{
        return false;
    }
}

bool check_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->lower_matrix.data[i]>0 && i != nnet->target){
            return false;
        }
    }
    return true;
}


bool check_min(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]<0 && i != nnet->target){
            return false;
        }
    }
    return true;
}

bool check_not_min(struct NNet *nnet, struct Interval *output){
	for(int i=0;i<nnet->outputSize;i++){
		if(output->lower_matrix.data[i]<0 && i != nnet->target){
			return true;
		}
	}
	return false;
}


bool check_not_min_p11(struct NNet *nnet, struct Interval *output){

    if(output->lower_matrix.data[0]<0)
        return true;

    return false;
}


bool check_max_constant1(struct NNet *nnet, struct Matrix *output){
    if(output->data[nnet->target]<0.5011){
        return false;
    }
    return true;
}


bool check_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return false;
        }
    }
    return true;
}


bool check_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return false;
        }
    }
    return true;
}


bool check_not_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return true;
        }
    }
    return false;
}


bool check_not_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return true;
        }
    }
    return false;
}


bool check_not_max_norm(struct NNet *nnet, struct Interval *output){
    float t = output->lower_matrix.data[nnet->target];
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]>t && i != nnet->target){
            return true;
        }
    }
    return false;
}


/*
 * Here is the function checking whether the output range always 
 * satisfies your customized safety property.
 */
bool check_functions(struct NNet *nnet, struct Interval *output){
    if (PROPERTY == 1){
        /*
         * You need to customize your own checking function
         * For instance, you can check whether the first output
         * is always the smallest. You can also check whether 
         * one of the output is always larger than 0.001, etc.
         */
    }

    return check_not_max(nnet, output);
}


/*
 * Here is the function checking whether the output range always 
 * satisfies your customized safety property but without output norm.
 * This is only used in network_test.c once for checking before splits.
 */
bool check_functions_norm(struct NNet *nnet, struct Interval *output){
    return check_not_max_norm(nnet, output);
}


/*
 * Here is the function checking whether the given concrete outupt 
 * violates your customized safety property.
 */
bool check_functions1(struct NNet *nnet, struct Matrix *output){

    if (PROPERTY == 1){
        /*
         * You need to customize your own checking function for adv
         * For instance, you can check whether the first output
         * is always the smallest. You can also check whether 
         * one of the output is always larger than 0.001, etc.
         */
    }

    return check_not_max1(nnet, output);
}










/*
 * Multithread function
 */
void *direct_run_check_conv_lp_thread(void *args){
    struct direct_run_check_conv_lp_args *actual_args = args;
    direct_run_check_conv_lp(actual_args->nnet, actual_args->input,\
                     actual_args->output_map,
                     actual_args->wrong_nodes, actual_args->wrong_node_length,\
                     actual_args->sigs,\
                     actual_args->equation_conv,\
                     actual_args->equation_conv_err,\
                     actual_args->err_row_conv,\
                     actual_args->target, actual_args->sig,\
                     actual_args->lp, actual_args->rule_num, actual_args->depth);
    return NULL;
}


void check_adv1(struct NNet* nnet, struct Matrix *adv){
    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop_conv(nnet, adv, &output);
    int is_adv = 0;
    // printMatrix(&output);
    is_adv = check_functions1(nnet, &output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(adv);
        printMatrix(&output);
        int adv_output = 0;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>0 && i != nnet->target){
                    adv_output = i;
            }
        }
        printf("%d ---> %d\n", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = true;
        pthread_mutex_unlock(&lock);
    }
}


int pop_queue(int *wrong_nodes, int *wrong_node_length){
    if(*wrong_node_length==0){
        printf("underflow\n");
        can_t_prove = true;
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

int max(float a, float b){
    return (a>b)?a:b;
}

int min(float a, float b){
    return (a<b)?a:b;
}


int sym_relu_lp(struct SymInterval *sInterval,
                    struct SymInterval *new_sInterval,
                    struct Interval *input,
                    struct NNet *nnet,
                    int layer, int err_row,
                    int*wrong_node_length, int *node_cnt,
                    int target, int sig, int *sigs,
                    lprec *lp, int *rule_num){

    float tempVal_upper=0.0, tempVal_lower=0.0;

    int inputSize = nnet->inputSize;
    struct Matrix weights = nnet->weights[layer];
    struct Matrix bias = nnet->bias[layer];
    //record the number of wrong nodes
    int wcnt = 0;

    for (int i=0; i < nnet->layerSizes[layer+1]; i++)
    {

        relu_bound(new_sInterval, nnet, input, i, layer, err_row,\
                    &tempVal_lower, &tempVal_upper);

        if(*node_cnt == target){
            if(sig==1){
                set_node_constraints(lp, (*new_sInterval->eq_matrix).data,\
                        i*(inputSize+1), rule_num, sig, inputSize);
            }
            else{
                set_node_constraints(lp, (*new_sInterval->eq_matrix).data,\
                        i*(inputSize+1), rule_num, sig, inputSize);
            }
        }
        
        //Perform ReLU relaxation
        //if(layer == 3) printf("%d %d %f %f\n", layer, i, tempVal_lower, tempVal_upper);

        // handle the nodes that are split
        //printf("splits for previous nodes\n");
        if(sigs[*node_cnt] == 0){
            tempVal_upper = 0;
        }
        else if(sigs[*node_cnt] == 1){
            tempVal_lower = 0;
        }

        relax_relu(nnet, new_sInterval, tempVal_lower, tempVal_upper, i,
            err_row, wrong_node_length, &wcnt);

        *node_cnt += 1;
    }

    return wcnt;

}


bool forward_prop_interval_equation_conv_lp(struct NNet *nnet,
                         struct Interval *input, bool *output_map,
                         int *sigs, float *equation_conv,
                         float *equation_conv_err, float err_row_conv,
                         int target, int sig,
                         lprec *lp, int *rule_num)
{
    int node_cnt=0;
    bool need_to_split = false;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    // equation is the temp equation for each layer

    ERR_NODE = 5000;
    float *equation_err = (float*)malloc(sizeof(float) *\
                            ERR_NODE*maxLayerSize);
    memset(equation_err, 0, sizeof(float)*ERR_NODE*maxLayerSize);
    float *new_equation_err = (float*)malloc(sizeof(float) *\
                            ERR_NODE*maxLayerSize);
    memset(new_equation_err, 0, sizeof(float)*ERR_NODE*maxLayerSize);

    float *equation = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    memset(equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
    float *new_equation = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    memset(new_equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);


    struct Matrix equation_matrix = {(float*)equation, inputSize+1, inputSize};
    struct Matrix new_equation_matrix = {(float*)new_equation, inputSize+1, inputSize};

    // The actual row number for index is ERR_NODE
    // but the row used for matmul is the true current error node err_row
    // This is because all the other lines are 0
    struct Matrix equation_err_matrix = {
                (float*)equation_err, ERR_NODE, inputSize
            };
    struct Matrix new_equation_err_matrix = {
                (float*)new_equation_err, ERR_NODE, inputSize
            };  

    struct SymInterval sInterval = {
                &equation_matrix, &equation_err_matrix
            };
    struct SymInterval new_sInterval = {
                &new_equation_matrix, &new_equation_err_matrix
            };

    
    for (int i=0; i < nnet->inputSize; i++)
    {
        equation[i*(inputSize+1)+i] = 1;
    }    

    //err_row is the number that is wrong before current layer
    int err_row=0;
    int wrong_node_length = 0;
    for (int layer = 0; layer<numLayers; layer++)
    {
        //printf("sig:%d, layer:%d\n",sig, layer );
        
        memset(new_equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_err,0,sizeof(float)*ERR_NODE*maxLayerSize);
        
        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];
        
        if (CHECK_ADV_MODE){
            if(layer>0 && nnet->layerTypes[layer]==0 &&\
                        nnet->layerTypes[layer-1]==1){
                memcpy(new_equation, equation_conv,\
                        sizeof(float)*(inputSize+1)*maxLayerSize);
                memcpy(new_equation_err, equation_conv_err,\
                        sizeof(float)*ERR_NODE*maxLayerSize);

                err_row = err_row_conv;
                wrong_node_length = err_row;
                equation_matrix.col = new_equation_matrix.col =\
                        nnet->layerSizes[layer+1];

                new_equation_err_matrix.row =\
                        equation_err_matrix.row = err_row;
                equation_err_matrix.col = new_equation_err_matrix.col =\
                        nnet->layerSizes[layer+1];
            }
            else if(nnet->layerTypes[layer]==1){
                node_cnt += nnet->layerSizes[layer+1];
                continue;
            }
            else{
                sym_fc_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
            }
        }
        else{
            if(nnet->layerTypes[layer] == 0){
                //printf("fc layer");
                sym_fc_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
            }
            else{
                //printf("conv layer\n");
                sym_conv_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
            }
        }
        
        if(layer<(numLayers-1)){
            // printf("relu layer\n");
            sym_relu_lp(&sInterval, &new_sInterval, input, nnet, layer,\
                        err_row, &wrong_node_length, &node_cnt,\
                        target, sig, sigs, lp, rule_num);
        }
        else{

            //printf("last layer\n");
            for (int i=0; i < nnet->layerSizes[layer+1]; i++){

                if(NEED_PRINT){
                    float tempVal_upper=0.0, tempVal_lower=0.0;
                    relu_bound(&new_sInterval, nnet, input, i, layer, err_row,\
                            &tempVal_lower, &tempVal_upper);
                    printf("target:%d, sig:%d, node:%d, l:%f, u:%f\n",\
                                target, sig, i, tempVal_lower, tempVal_upper);
                }
                

                if(i!=nnet->target){
                    float upper_err=0, lower_err=0;
                    for(int k=0;k<inputSize+1;k++){
                        new_equation[k+i*(inputSize+1)] -=\
                                new_equation[k+nnet->target*(inputSize+1)]; 
                    }
                    //if(i < nnet->target){
                    //    new_equation[inputSize+i*(inputSize+1)] -= bias.data[nnet->target];
                    //}
                    
                    for(int err_ind=0;err_ind<err_row;err_ind++){
                        new_equation_err[err_ind+i*ERR_NODE] -=\
                                    new_equation_err[err_ind+nnet->target*ERR_NODE];
                        if(new_equation_err[err_ind+i*ERR_NODE]>0){
                            upper_err += new_equation_err[err_ind+i*ERR_NODE];
                        }
                        else{
                            lower_err += new_equation_err[err_ind+i*ERR_NODE];
                        }
                    }
                    new_equation[inputSize+i*(inputSize+1)] += upper_err;

                    //gettimeofday(&start, NULL);
                    float upper = 0.0;
                    float input_prev[inputSize];
                    struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                    memset(input_prev, 0, sizeof(float)*inputSize);
                    float o[outputSize];
                    struct Matrix output_matrix = {o, outputSize, 1};
                    memset(o, 0, sizeof(float)*outputSize);
                    if(output_map[i]){
                        if(!set_output_constraints(lp, new_equation, i*(inputSize+1),\
                                    rule_num, inputSize, MAX, &upper, input_prev)){
                            need_to_split = true;
                            output_map[i] = true;
                            if(NEED_PRINT){
                                printf("target:%d, sig:%d, node:%d--Objective value: %f\n",\
                                            target, sig, i, upper);
                            }
                            check_adv1(nnet, &input_prev_matrix);
                            if(adv_found){
                                return 0;
                            }
                        }
                        else{
                            output_map[i] = false;
                            if(NEED_PRINT){
                                printf("target:%d, sig:%d, node:%d--unsat\n",\
                                            target, sig, i);
                            }
                        }
                    }

                }

                node_cnt++;
            }
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

    //printf("sig:%d, need_to_split:%d\n",sig, need_to_split );

    free(equation);
    free(new_equation);
    free(equation_err);
    free(new_equation_err);
    return need_to_split;
}


bool direct_run_check_conv_lp(struct NNet *nnet, struct Interval *input,
                     bool *output_map, int *wrong_nodes, int *wrong_node_length,
                     int *sigs, float *equation_conv, float *equation_conv_err,
                     float err_row_conv, int target, int sig, lprec *lp,
                     int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return false;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return false;
    }
    pthread_mutex_unlock(&lock);

    if(depth<=3){
        solve(lp);
    }

    bool isOverlap = forward_prop_interval_equation_conv_lp(nnet, input,\
                            output_map, \
                            sigs, equation_conv, equation_conv_err,\
                            err_row_conv, target, sig,\
                            lp, rule_num);

    //printf("sig:%d, i:%d\n",sig, isOverlap );
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
        if(NEED_PRINT)
            printf("depth:%d, sig:%d Need to split!\n\n", depth, sig);
        isOverlap = split_interval_conv_lp(nnet, input, output_map,
                         wrong_nodes, wrong_node_length, sigs,\
                         equation_conv, equation_conv_err, err_row_conv,\
                         lp, rule_num, depth);
    }
    else{
        if(!adv_found)
            if(NEED_PRINT) 
                printf("depth:%d, sig:%d, UNSAT, great!\n\n", depth, sig);
            pthread_mutex_lock(&lock);
                avg_depth -= (avg_depth) / AVG_WINDOW;
                avg_depth += depth / AVG_WINDOW;
            pthread_mutex_unlock(&lock);
    }
    return isOverlap;
}


bool split_interval_conv_lp(struct NNet *nnet, struct Interval *input,
                     bool *output_map, int *wrong_nodes, int *wrong_node_length,
                     int *sigs, float *equation_conv, float *equation_conv_err,
                     float err_row_conv, lprec *lp, int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return false;
    }
    
    if(depth>=MAX_DEPTH){
        can_t_prove = true;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return false;
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

    int target = pop_queue(wrong_nodes, wrong_node_length);
    // printf("%d, %d\n", wrong_nodes[0], wrong_nodes[1]);
    int sig = 0;
    bool isOverlap1 = false;
    bool isOverlap2 = false;

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

    bool output_map1[outputSize];
    bool output_map2[outputSize];
    memcpy(output_map1, output_map, sizeof(bool)*outputSize);
    memcpy(output_map2, output_map, sizeof(bool)*outputSize);

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
    if((depth <= avg_depth - MIN_DEPTH_PER_THREAD) &&\
            (count<=MAX_THREAD) && !NEED_FOR_ONE_RUN) {
        pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;
        struct direct_run_check_conv_lp_args args1 = {
                            nnet, input, output_map1,
                            wrong_nodes1, &wrong_node_length1, sigs1,\
                            equation_conv, equation_conv_err, err_row_conv,\
                            target, sig1,\
                            lp1, &rule_num1, depth
                        };

        struct direct_run_check_conv_lp_args args2 = {
                            nnet, input, output_map2,
                            wrong_nodes2, &wrong_node_length2, sigs2,\
                            equation_conv, equation_conv_err, err_row_conv,\
                            target, sig2,\
                            lp2, &rule_num2, depth
                        };

        pthread_create(&workers1, NULL,\
                direct_run_check_conv_lp_thread, &args1);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, NULL,\
                direct_run_check_conv_lp_thread, &args2);
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

        isOverlap1 = false;
        isOverlap2 = false;

    }
    else{
        pthread_mutex_unlock(&lock);
        isOverlap1 = direct_run_check_conv_lp(nnet, input,\
                            output_map1,\
                            wrong_nodes1, &wrong_node_length1, sigs1,\
                            equation_conv, equation_conv_err, err_row_conv,\
                            target, sig1,\
                            lp1, &rule_num1, depth);

        isOverlap2 = direct_run_check_conv_lp(nnet, input,\
                            output_map2,\
                            wrong_nodes2, &wrong_node_length2, sigs2,\
                            equation_conv, equation_conv_err, err_row_conv,\
                            target, sig2,\
                            lp2, &rule_num2, depth);
    }

    delete_lp(lp1);
    delete_lp(lp2);

    bool result = isOverlap1 || isOverlap2;
    depth --;

    if(!result && depth<=PROGRESS_DEPTH){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            fprintf(stderr, " %d/%d ",\
                    progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
        pthread_mutex_unlock(&lock);
    }

    return result;
}
