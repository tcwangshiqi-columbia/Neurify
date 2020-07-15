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

#include <cblas.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"

void sig_handler(int signo)
{
    if (signo == SIGQUIT){
        printf("progress: %d/1024\n", progress);

    }
}

char* config_args(int argc, char *argv[]){
    char *FULL_NET_PATH = NULL;
    if(argc>8 || argc<3) {
        printf("please specify a network\n");
        printf("./network_test [property] [network]"
                " [print] [test for one run] [check mode]"
                " [max depth] [norm input]\n");
        exit(1);
    }
    for(int i=1;i<argc;i++){
        if(i==1){
            PROPERTY = atoi(argv[i]); 
            if(PROPERTY<0){
                printf("Wrong input depth");
                exit(1);
            } 
        }
        if(i==2){
            FULL_NET_PATH = strdup(argv[i]);
        }
        if(i==3){
            NEED_PRINT = atoi(argv[i]);
            if(NEED_PRINT != 0 && NEED_PRINT!=1){
                printf("Wrong print");
                exit(1);
            }
        }
        if(i==4){
            NEED_FOR_ONE_RUN = atoi(argv[i]); 
            if(NEED_FOR_ONE_RUN != 0 && NEED_FOR_ONE_RUN != 1){
                printf("Wrong test for one run");
                exit(1);
            }
        }
        if(i==5){
            if(atoi(argv[i])==0){
                // Regular mode
                CHECK_ADV_MODE = 0;
                MAX_DEPTH = 10;
            }
            if(atoi(argv[i])==1){
                // Only check for adv
                CHECK_ADV_MODE = 1;
                MAX_DEPTH = 10;
                printf("CHECK ADV MODE with MAX_DEPTH 10 by default\n");
            }
        }
        if(i==6){
            MAX_DEPTH = atoi(argv[i]);
        }
        if(i==7){
            if (NORM_INPUT!=0 && NORM_INPUT!=1){
                printf("NORM INPUT only be 0/1");
                exit(1);
            }
            NORM_INPUT = atoi(argv[i]);
        }
    }

    printf("MAX_DEPTH: %d, NORM_INPUT: %d\n", MAX_DEPTH, NORM_INPUT);
    return FULL_NET_PATH;
}


int main( int argc, char *argv[]){
    char *FULL_NET_PATH = config_args(argc, argv);
    printf("%s\n", FULL_NET_PATH);
    
    openblas_set_num_threads(1);

    srand((unsigned)time(NULL));
    double time_spent = 0;
    double total_time_spent = 0;

    int image_start = 0;
    int image_length = 0;
    if(PROPERTY == 0){
        image_length = 1000;
        image_start = 0;
        INF = 10;
    }
    else if(PROPERTY==1){
        /*
         * Customize your own property you want to verify
         * For instance, you can check whether the first ouput is
         * always the smallest or the second output is always
         * less than 0.01
         * For each property, you need to change (1) the dataset
         * that you want to load in nnet.c; (2) the check_function
         * and check_function1 in split.c.
         */
    }
    else{
        image_length = 1;
        image_start = 0;
        INF = 0;
    }

    int adv_num = 0;
    int non_adv = 0;
    int no_prove = 0;
    int can_t_prove_list[image_length];
    memset(can_t_prove_list, 0, sizeof(int)*image_length);

    float avg_wrong_length = 0.0;
    
    for(int img=image_start; img<image_start+image_length; img++){
        gettimeofday(&start, NULL);

        adv_found = false;
        analysis_uncertain = false;
        printf("start loading network\n");
        struct NNet* nnet = load_conv_network(FULL_NET_PATH, img);
        printf("done loading network\n");

        int numLayers    = nnet->numLayers;
        int inputSize    = nnet->inputSize;
        int outputSize   = nnet->outputSize;
        int maxLayerSize = nnet->maxLayerSize;

        float u[inputSize+1], l[inputSize+1], input_prev[inputSize];
        struct Matrix input_prev_matrix = {input_prev, 1, inputSize};

        struct Matrix input_upper = {u,1,nnet->inputSize+1};
        struct Matrix input_lower = {l,1,nnet->inputSize+1};
        struct Interval input_interval = {input_lower, input_upper};

        initialize_input_interval(nnet, img, inputSize, input_prev, u, l);

        if(NORM_INPUT){
            normalize_input(nnet, &input_prev_matrix);
            normalize_input_interval(nnet, &input_interval);
        }

        float o[nnet->outputSize];
        struct Matrix output = {o, outputSize, 1};
        
        float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
        struct Interval output_interval = {
                    (struct Matrix){o_lower, outputSize, 1},
                    (struct Matrix){o_upper, outputSize, 1}
                };

        printf("running image %d with network %s\n", img, FULL_NET_PATH);
        printf("Infinite Norm: %f\n", INF);
        //printMatrix(&input_upper);
        //printMatrix(&input_lower);

        for(int i=0;i<inputSize;i++){

            if(input_interval.upper_matrix.data[i]<\
                        input_interval.lower_matrix.data[i]){
                printf("wrong input!\n");
                exit(0);
            }
        }

        
        //forward_prop(nnet, &input_prev_matrix, &output);
        if(inputSize<10){
            printf("concrete input:");
            printMatrix(&input_prev_matrix);
        }
        evaluate_conv(nnet, &input_prev_matrix, &output);
        printf("concrete output:");
        printMatrix(&output);

        bool is_overlap = false;

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

		avg_wrong_length += wrong_node_length; 

        printf("total wrong nodes: %d, wrong nodes in "\
                    "fully connected layers: %d\n", wrong_node_length,\
                    full_wrong_node_length );
        /*
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
            if(full_wrong_node_length == 0) {
                printf("Not implemented: At least one node needs to be able to be split to " \
                    "test the LP. \n");
                exit(1);
            }
            if(CHECK_ADV_MODE){
                printf("Check Adv Mode (CHECK_ADV_MODE)\n");
                for (int n=0;n<full_wrong_node_length;n++){
                    wrong_nodes_map[n] = wrong_nodes_map[err_row_conv+n];
                }
                wrong_node_length = full_wrong_node_length;
            }
            else{
                printf("Regular Mode (No CHECK_ADV_MODE)\n");
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
        time_spent = ((float)(finish.tv_sec-start.tv_sec)*1000000 +\
                (float)(finish.tv_usec-start.tv_usec)) / 1000000;
        total_time_spent += time_spent;

        if(!is_overlap && !adv_found && !analysis_uncertain){
            if (CHECK_ADV_MODE){
                printf("no adv found\n");
                can_t_prove_list[no_prove] = img;
                no_prove ++;
            }
            else{
                printf("No adv!\n");
                non_adv ++;
            }
        }
        else if(adv_found){
            adv_num ++;
        }
        else{
            can_t_prove_list[no_prove] = img;
            no_prove ++;
            printf("can't prove!\n");
        }

        printf("Current analysis result: %d adv, %d non-adv, %d undetermined \n",
          adv_num, non_adv, no_prove);
        printf("time: %f \n\n", time_spent);
        destroy_conv_network(nnet);

        free(equation_conv);
        free(equation_conv_err);
        delete_lp(lp);

    }

    avg_wrong_length /= (float)image_length;

    printf("Final analysis result: %d adv, %d non-adv, %d undetermined \n", 
        adv_num, non_adv, no_prove);
    printf("avg wrong node length:%f\n", avg_wrong_length);
    printf("Total time: %f \n\n", total_time_spent);
    if(no_prove>0){
        printf("images that have not been proved:\n");
        for(int ind=0;ind<no_prove;ind++){
            printf("%d ", can_t_prove_list[ind]);
        }
        printf("\n");
    }
    
    
}


