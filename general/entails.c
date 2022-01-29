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

#include <openblas/cblas.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"
#include "entails.h"


void sig_handler(int signo)
{
    if (signo == SIGQUIT){
        //printf("progress: %d/1024\n", progress);

    }
}

// h: (array of int) indexes of elements in hitting set
// input:  (array of float) the concrete input
// eps: (float) the value of epsilon we want to verify within
int entails( int *h, int h_size, float *input, int input_size, float *u_bounds, float *l_bounds, char* network_path){
    openblas_set_num_threads(1);

    PROPERTY = 0;
    int i;
    printf("in len %d\n",input_size);
    printf(" \n");
    for (i=0;i < input_size ;i++) {
            printf("%lf ",u_bounds[i]);
            fflush(stdout);
    }
    printf(" \n");
    for (i=0;i < input_size; i++) {
            printf("%lf ",l_bounds[i]);
            fflush(stdout);
    }
    printf("\n\n");
    fflush(stdout);

    adv_found = false;
    analysis_uncertain = false;
    printf("start loading network\n");
    struct NNet* nnet = load_conv_network(network_path, input);
    printf("done loading network\n");

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    float u[inputSize+1], l[inputSize+1];
    struct Matrix input_matrix = {input, 1, inputSize};

    struct Matrix input_upper = {u,1,nnet->inputSize+1};
    struct Matrix input_lower = {l,1,nnet->inputSize+1};
    struct Interval input_interval = {input_lower, input_upper};

    initialize_input_interval(nnet, input, input_size, h, h_size, u, l, u_bounds, l_bounds);

    float o[nnet->outputSize];
    struct Matrix output = {o, outputSize, 1};
    
    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {
                (struct Matrix){o_lower, outputSize, 1},
                (struct Matrix){o_upper, outputSize, 1}
            };

    printf("running input %d with network %s\n", 0, network_path);
    //printf("Infinite Norm: %f\n", eps);
    printMatrix(&input_upper);
    printMatrix(&input_lower);

    for(int i=0;i<inputSize;i++){

       // printf("upper: %f, lower: %f",input_interval.upper_matrix.data[i],input_interval.lower_matrix.data[i]);
        if(input_interval.upper_matrix.data[i]<\
                    input_interval.lower_matrix.data[i]){
            printf("wrong input!\n");
            exit(0);
        }
    }
    
    //printf("--------------\n");
    evaluate_conv(nnet, &input_matrix, &output);
    //printf("concrete output:");
    //printMatrix(&output);

    bool is_overlap = false;
    int adv_num = 0;
    int non_adv = 0;
    int no_prove = 0;

    int total_nodes = 0;
    for(int layer=1;layer<numLayers;layer++){
        total_nodes += nnet->layerSizes[layer];
    }
    int wrong_nodes_map[total_nodes];
    memset(wrong_nodes_map,0,sizeof(int)*total_nodes);

    float grad[total_nodes];
    memset(grad, 0, sizeof(float)*total_nodes);

    int wrong_node_length = 0;
    int full_wrong_node_length = 0;
    
    ERR_NODE = 5000;
    // the equation of last convolutional layer 
    float *equation_conv = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *equation_conv_err = (float*)malloc(sizeof(float) *\
                            ERR_NODE*maxLayerSize);

    int err_row_conv = 0;

    forward_prop_interval_equation_linear_conv(nnet, &input_interval,\
                         &output_interval,\
                         grad, wrong_nodes_map, &wrong_node_length,\
                         &full_wrong_node_length,\
                         equation_conv, equation_conv_err, &err_row_conv);

    printf("One shot approximation:\n");
    printf("upper_matrix:");
    printMatrix(&output_interval.upper_matrix);
    printf("lower matrix:");
    printMatrix(&output_interval.lower_matrix);

    for(int i = 0; i < outputSize; i++) {
        if(output_interval.upper_matrix.data[i] < output.data[i] ||
           output_interval.lower_matrix.data[i] > output.data[i]) {
            printf("Invalid approximation \n");
            exit(1);
        }
    }


    /*
    printf("total wrong nodes: %d, wrong nodes in "\
                "fully connected layers: %d\n", wrong_node_length,\
                full_wrong_node_length );
    for(int w=0;w<wrong_node_length;w++){
        printf("%d,",wrong_nodes_map[w]);
    }
    */

    bool output_map[outputSize];
    for(int oi=0;oi<outputSize;oi++){
        if(output_interval.upper_matrix.data[oi]>\
                output_interval.lower_matrix.data[nnet->target] &&\
                oi!=nnet->target){
            output_map[oi] = true;
        }
        else{
            output_map[oi] = false;
        }
    }
    is_overlap = check_functions_norm(nnet, &output_interval);

    lprec *lp;
    
    int rule_num = 0;
    int Ncol = inputSize;
    lp = make_lp(0, Ncol);
    set_verbose(lp, IMPORTANT);
    
    set_input_constraints(&input_interval, lp, &rule_num, inputSize);
    set_presolve(lp, PRESOLVE_LINDEP, get_presolveloops(lp));
    //write_LP(lp, stdout);

    if(is_overlap){
        //if(full_wrong_node_length == 0) {
        if(wrong_node_length == 0) {
            printf("Not implemented: At least one node needs to be able to be split to " \
                "test the LP. \n");
            return 2;
        }
        if(CHECK_ADV_MODE){
            //printf("Check Adv Mode (CHECK_ADV_MODE)\n");
            for (int n=0;n<full_wrong_node_length;n++){
                wrong_nodes_map[n] = wrong_nodes_map[err_row_conv+n];
            }
            wrong_node_length = full_wrong_node_length;
        }
        else{
            //printf("Regular Mode (No CHECK_ADV_MODE)\n");
        }

        int sigs[total_nodes];
        memset(sigs, -1, sizeof(int)*total_nodes);

        // split
        int depth = 0;
        is_overlap = split_interval_conv_lp(nnet, &input_interval,\
                            output_map,\
                            grad, wrong_nodes_map, &wrong_node_length, sigs,\
                            equation_conv, equation_conv_err,\
                            err_row_conv,\
                            lp, &rule_num, depth);
    }

    //write_LP(lp, stdout);
    //is_overlap = check_functions(nnet, &output_interval);
    
    gettimeofday(&finish, NULL);

    if(!is_overlap && !adv_found && !analysis_uncertain){
        if (CHECK_ADV_MODE){
            //printf("no adv found\n");
            no_prove ++;
        }
        else{
            //printf("No adv!\n");
            non_adv ++;
        }
    }
    else if(adv_found){
        adv_num ++;
    }
    else{
        //printf("can't prove!\n");
    }

    //printf("Current analysis result: %d adv, %d non-adv, %d undetermined \n",
    //  adv_num, non_adv, no_prove);
    destroy_conv_network(nnet);

    free(equation_conv);
    free(equation_conv_err);
    delete_lp(lp);

    // only need to know existence of an adv so return 1 if adv, else 0
    if (adv_found) {
        //printf("Adv found.\n");
        return 1; 
    } else {
        //printf("NO adv found.\n");
        return 0;
    }
    
}


