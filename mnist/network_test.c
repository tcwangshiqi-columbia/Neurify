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


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"

//extern int thread_tot_cnt;

void sig_handler(int signo)
{
    if (signo == SIGQUIT){
        printf("progress: %d/1024\n", progress);

    }
}

int main( int argc, char *argv[]){

    char *FULL_NET_PATH;
    if(argc>6 || argc<3) {
        printf("please specify a network\n");
        printf("./network_test [property] [network] [print] [test for one run] [check mode]\n");
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
            FULL_NET_PATH = argv[i];
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
            if(NEED_FOR_ONE_RUN != 0 && NEED_FOR_ONE_RUN!=1){
                printf("Wrong test for one run");
                exit(1);
            }
        }
        if(i==5){
            if(atoi(argv[i])==0){
                CHECK_ADV_MODE = 0;
                PARTIAL_MODE = 0;
            }
            if(atoi(argv[i])==1){
                CHECK_ADV_MODE = 1;
                PARTIAL_MODE = 0;
            }
            if(atoi(argv[i])==2){
                CHECK_ADV_MODE = 0;
                PARTIAL_MODE = 1;
            }
            if(atoi(argv[i])==3){
                ACCURATE_MODE = 0;
            }
        }
    }
    openblas_set_num_threads(1);

    srand((unsigned)time(NULL));
    double time_spent;
    int i,j,layer;

    int image_start, image_length;
    if(PROPERTY == 0){
        /*
         * choose all pixels to intervals
         * 69(8406),71(114.10), 82(217) are three hard-provable images
         * 69(321),71(31.73),82(58.14)
         * 50: 82(2348.40)
         */
        image_length = 100;
        image_start = 0;
        INF = 15;
    }
    else if(PROPERTY == 1){
        /*
         * choose 200 most uninfluencial pixels to interval
         */
        image_length = 20;
        image_start = 0;
        INF = 25;
    }
    else if(PROPERTY == 2){
        /*
         * choose 200 most influencial pixels to interval
         */
        image_length = 20;
        image_start = 0;
        INF = 25;
    }
    else if(PROPERTY == 3){
        /*
         * randomly choose 200 pixels to interval
         */
        image_length = 1;
        image_start = 10;
        INF = 25;
    }
    else if(PROPERTY ==4){
        //conv
        //INF = 5, img=9
        image_length = 1;
        image_start = 0;
        INF = 25;
        //INF = 0;
    }
    else if(PROPERTY == 11){
        // fmnist dataset
        image_length = 100;
        image_start = 0;
        INF = 25;
    }
    else if(PROPERTY == 54){
        // L1 for convolutional
        image_length = 10;
        image_start = 0;
        INF = 25;
    }
    else if(PROPERTY == 200){
        //drebin
        image_length = 1;
        image_start = 1;
        INF = 0.08;
    }
    else if(PROPERTY == 500){
        //self-driving
        image_length = 1;
        image_start = 0;
        INF = 25;
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
    
    for(int img_ind=0; img_ind<image_length;img_ind++){
        int img = image_start + img_ind;
        adv_found=0;
        can_t_prove=0;
        printf("start load network\n");
        struct NNet* nnet = load_conv_network(FULL_NET_PATH, img);
        printf("done load network\n");

        int numLayers    = nnet->numLayers;
        int inputSize    = nnet->inputSize;
        int outputSize   = nnet->outputSize;
        int maxLayerSize = nnet->maxLayerSize;

        float u[inputSize], l[inputSize], input_prev[inputSize];
        struct Matrix input_prev_matrix = {input_prev, 1, inputSize};

        struct Matrix input_upper = {u,1,nnet->inputSize};
        struct Matrix input_lower = {l,1,nnet->inputSize};
        struct Interval input_interval = {input_lower, input_upper};

        initialize_input_interval(nnet, img, inputSize, input_prev, u, l);
        if(PROPERTY<500){
            normalize_input(nnet, &input_prev_matrix);
            normalize_input_interval(nnet, &input_interval);
        }
        float grad_upper[inputSize], grad_lower[inputSize];
        struct Interval grad_interval = {(struct Matrix){grad_upper, 1, inputSize},
                                         (struct Matrix){grad_lower, 1, inputSize}};

        float o[nnet->outputSize];
        struct Matrix output = {o, outputSize, 1};
        
        float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
        struct Interval output_interval = {(struct Matrix){o_lower, outputSize, 1},
                                           (struct Matrix){o_upper, outputSize, 1}};

        int n = 0;
        int feature_range_length = 0;
        int split_feature = -1;
        printf("running image %d with network %s\n", img, FULL_NET_PATH);
        printf("Infinite Norm: %f\n", INF);
        //printMatrix(&input_upper);
        //printMatrix(&input_lower);
        for(int i=0;i<inputSize;i++){
            if(input_interval.upper_matrix.data[i]<input_interval.lower_matrix.data[i]){
                printf("wrong input!\n");
                exit(0);
            }
            if(input_interval.upper_matrix.data[i]!=input_interval.lower_matrix.data[i]){
                n++;
            }
        }
        feature_range_length = n;
        int *feature_range = (int*)malloc(n*sizeof(int));
        for(int i=0, n=0;i<nnet->inputSize;i++){
            if(input_interval.upper_matrix.data[i]!=input_interval.lower_matrix.data[i]){
                feature_range[n] = i;
                n++;
            }
        }

        float *equation_upper = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
        float *equation_lower = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
        float *new_equation_upper = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);
        float *new_equation_lower = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);

        float *equation_input_lower = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*nnet->layerSizes[1]);
        float *equation_input_upper = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*nnet->layerSizes[1]);

        struct Interval equation_inteval = {
            (struct Matrix){(float*)equation_lower, inputSize+1, nnet->layerSizes[1]},
            (struct Matrix){(float*)equation_upper, inputSize+1, nnet->layerSizes[1]}
        };

        float *equation = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
        float *new_equation = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);

        
        //forward_prop(nnet, &input_prev_matrix, &output);
        if(inputSize<10){
            printf("concrete input:");
            printMatrix(&input_prev_matrix);
        }
        evaluate_conv(nnet, &input_prev_matrix, &output);
        printf("concrete output:");
        printMatrix(&output);

        gettimeofday(&start, NULL);
        int isOverlap = 0;

        int wrong_node_length = 0; 
        int full_wrong_node_length = 0;
        for(int layer=1;layer<numLayers;layer++){
            wrong_node_length += nnet->layerSizes[layer];
        }
        int wrong_nodes[wrong_node_length];
        float wrong_up_s_up[wrong_node_length];
        float wrong_up_s_low[wrong_node_length];
        float wrong_low_s_up[wrong_node_length];
        float wrong_low_s_low[wrong_node_length];
        
        memset(wrong_nodes,0,sizeof(int)*wrong_node_length);
        memset(wrong_up_s_up,0,sizeof(float)*wrong_node_length);
        memset(wrong_up_s_low,0,sizeof(float)*wrong_node_length);
        memset(wrong_low_s_up,0,sizeof(float)*wrong_node_length);
        memset(wrong_low_s_low,0,sizeof(float)*wrong_node_length);

        int sigs[wrong_node_length];
        memset(sigs, -1, sizeof(int)*wrong_node_length);

        float grad[wrong_node_length];
        memset(grad, 0, sizeof(float)*wrong_node_length);

        wrong_node_length = 0;


        //forward_prop_interval_equation_linear2(nnet, &input_interval, &output_interval,\
                                         grad, equation_upper, equation_lower,\
                                         new_equation_upper, new_equation_lower,\
                                         wrong_nodes, &wrong_node_length,\
                                         wrong_up_s_up, wrong_up_s_low,\
                                         wrong_low_s_up, wrong_low_s_low);
        ERR_NODE = 5000;
        //wrong_node_length = 0;
        float *equation_err = (float*)malloc(sizeof(float) *\
                                ERR_NODE*maxLayerSize);
        float *new_equation_err = (float*)malloc(sizeof(float) *\
                                ERR_NODE*maxLayerSize);
        // the equation of last convolutional layer 
        float *equation_conv = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
        float *equation_conv_err = (float*)malloc(sizeof(float) *\
                                ERR_NODE*maxLayerSize);

        int err_row_conv = 0;

        forward_prop_interval_equation_linear_conv(nnet, &input_interval, &output_interval,\
                                         grad, equation, equation_err,\
                                         new_equation, new_equation_err,\
                                         wrong_nodes, &wrong_node_length,\
                                         &full_wrong_node_length,\
                                         equation_conv, equation_conv_err, &err_row_conv);

        printf("One shot approximation:\n");
        printf("upper_matrix:");
        printMatrix(&output_interval.upper_matrix);
        printf("lower matrix:");
        printMatrix(&output_interval.lower_matrix);
		
        sort(grad, full_wrong_node_length, wrong_nodes);
		
		sort_layers(nnet->numLayers, nnet->layerSizes, full_wrong_node_length, wrong_nodes);
		avg_wrong_length += wrong_node_length; 

        printf("total wrong nodes: %d, wrong nodes in fully connect layers: %d\n", wrong_node_length, full_wrong_node_length );
        /*
		for(int w=0;w<full_wrong_node_length;w++){
            printf("%d\n",wrong_nodes[w] );
        }
		*/ 
        int output_map[outputSize];
        for(int oi=0;oi<outputSize;oi++){
            if(output_interval.upper_matrix.data[oi]>0 && oi!=nnet->target){
                output_map[oi]=1;
            }
            else{
                output_map[oi]=0;
            }
        }

        isOverlap = check_functions_norm(nnet, &output_interval);
        lprec *lp;
        
        int rule_num = 0;
        int Ncol = inputSize;
        REAL row[Ncol+1];
        lp = make_lp(0, Ncol);
        set_verbose(lp, IMPORTANT);
        
        set_input_constraints(&input_interval, lp, &rule_num);
        set_presolve(lp, PRESOLVE_LINDEP, get_presolveloops(lp));
        //write_LP(lp, stdout);
        int target = 0;
        int sig = 0;

        gettimeofday(&start, NULL);
        int depth = 0;
        if(isOverlap){
            if(CHECK_ADV_MODE){
                printf("CHECK_ADV_MODE\n");
            }
            else{
                printf("No CHECK_ADV_MODE\n");
            }
            //isOverlap = split_interval_lp(nnet, &input_interval,\
                         &grad_interval,output_map,\
                         equation_upper, equation_lower,\
                         new_equation_upper, new_equation_lower,\
                         wrong_nodes, &wrong_node_length, sigs,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         lp, &rule_num, depth);
            // one shot
            // isOverlap = forward_prop_interval_equation_conv_lp(nnet, &input_interval, output_map,\
                                 equation, equation_err,\
                                 new_equation, new_equation_err,\
                                 sigs, equation_conv, equation_conv_err, err_row_conv,\
                                 target, sig,\
                                 lp, &rule_num);
            // split
            isOverlap = split_interval_conv_lp(nnet, &input_interval, output_map,\
                                                equation, equation_err,\
                                                new_equation, new_equation_err,\
                                                wrong_nodes, &full_wrong_node_length, sigs,\
                                                equation_conv, equation_conv_err, err_row_conv,\
                                                lp, &rule_num, depth);
        }

        //write_LP(lp, stdout);
        //isOverlap = check_functions(nnet, &output_interval);
        
        gettimeofday(&finish, NULL);
        time_spent = ((float)(finish.tv_sec-start.tv_sec)*1000000 +\
                     (float)(finish.tv_usec-start.tv_usec)) / 1000000;

        if(isOverlap==0 && adv_found == 0 && !can_t_prove){
            printf("No adv!\n");
            non_adv ++;
        }
        else if(adv_found){
            adv_num ++;
        }
        else{
            can_t_prove_list[no_prove] = img;
            no_prove ++;
            printf("can't prove!\n");
        }

        printf("%d %d %d\n", adv_num, non_adv, no_prove);
        printf("time: %f \n\n", time_spent);
        destroy_conv_network(nnet);
        free(feature_range);

        free(equation_upper);
        free(equation_lower);
        free(new_equation_upper);
        free(new_equation_lower);
        free(equation_input_upper);
        free(equation_input_lower);
        free(equation);
        free(new_equation);
        free(equation_err);
        free(new_equation_err);
        delete_lp(lp);

    }

    avg_wrong_length /= (float)image_length;

    printf("adv: %d, non-adv: %d, not_proved: %d\n", adv_num, non_adv, no_prove);
    printf("avg wrong node length:%f\n", avg_wrong_length);
    if(no_prove>0){
        printf("image that have not been proved:\n");
        for(int ind=0;ind<no_prove;ind++){
            printf("%d ", can_t_prove_list[ind]);
        }
        printf("\n");
    }
    
    
}


