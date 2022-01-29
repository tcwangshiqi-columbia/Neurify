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

#include "nnet.h"


char *LOG_FILE = "logs/log.txt";
int ERR_NODE=10;
int CHECK_ADV_MODE = 0;
int PROPERTY;
struct timeval start,finish,last_finish;
//FILE *fp;

//Take in a .nnet filename with path and load the network from the file
//Inputs:  filename - const char* that specifies the name and path of file
//Outputs: void *   - points to the loaded neural network
struct NNet *load_conv_network(const char* filename, float *input)
{
    //Load file and check if it exists
    FILE *fstream = fopen(filename,"r");
    
    if (fstream == NULL)
    {
        printf("Wrong network!\n");
        exit(1);
    }
    //Initialize variables
    int bufferSize = 650000;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;

    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));

    //memset(nnet, 0, sizeof(struct NNet));
    //Read int parameters of neural network

    line=fgets(buffer,bufferSize,fstream);
    while (strstr(line, "//")!=NULL)
        line=fgets(buffer,bufferSize,fstream); //skip header lines
    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    printf("max layer size %d\n",nnet->maxLayerSize);


    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (int i = 0; i<((nnet->numLayers)+1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }
    //Load Min and Max values of inputs
    nnet->min = 0.0;
    nnet->max = 1.0;
    
    nnet->layerTypes = (int*)malloc(sizeof(int)*nnet->numLayers);
    nnet->convLayersNum = 0;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (int i = 0; i<nnet->numLayers; i++)
    {
        nnet->layerTypes[i] = atoi(record);
        if(nnet->layerTypes[i]==1){
            nnet->convLayersNum++;
        }
        record = strtok(NULL,",\n");
    }
    //initial convlayer parameters
    nnet->convLayer = (int**)malloc(sizeof(int *)*nnet->convLayersNum);
    for(int i = 0; i < nnet->convLayersNum; i++){
        nnet->convLayer[i] = (int*)malloc(sizeof(int)*5);
    }

    for(int cl=0;cl<nnet->convLayersNum;cl++){
        line = fgets(buffer,bufferSize,fstream);
        record = strtok(line,",\n");
        for (int i = 0; i<5; i++){
            nnet->convLayer[cl][i] = atoi(record);

            //printf("%d,", nnet->convLayer[cl][i]);

            record = strtok(NULL,",\n");
        }

        printf("\n");
    }

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only one number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = (float****)malloc(sizeof(float *)*(nnet->numLayers));
    for (int layer = 0; layer<nnet->numLayers; layer++){
        if(nnet->layerTypes[layer]==0){
            nnet->matrix[layer] = (float***)malloc(sizeof(float *)*2);
            nnet->matrix[layer][0] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
            nnet->matrix[layer][1] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
            for (int row = 0; row < nnet->layerSizes[layer+1]; row++){
                nnet->matrix[layer][0][row] = (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
                nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
            }
        }
    }

    nnet->conv_matrix = (float****)malloc(sizeof(float *)*nnet->convLayersNum);
    for(int layer = 0; layer < nnet->convLayersNum; layer++){
        int out_channel = nnet->convLayer[layer][0];
        int in_channel = nnet->convLayer[layer][1];
        int kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
        nnet->conv_matrix[layer]=(float***)malloc(sizeof(float*)*out_channel);
        for(int oc=0;oc<out_channel;oc++){
            nnet->conv_matrix[layer][oc] = (float**)malloc(sizeof(float*)*in_channel);
            for(int ic=0;ic<in_channel;ic++){
                nnet->conv_matrix[layer][oc][ic] = (float*)malloc(sizeof(float)*kernel_size);
            }

        }
    }

    nnet->conv_bias = (float**)malloc(sizeof(float*)*nnet->convLayersNum);
    for(int layer = 0; layer < nnet->convLayersNum; layer++){
        int out_channel = nnet->convLayer[layer][0];
        nnet->conv_bias[layer] = (float*)malloc(sizeof(float)*out_channel);
    }
    
    int layer = 0;
    int param = 0;
    int i=0;
    int j=0;
    char *tmpptr=NULL;

    int oc=0, ic=0, kernel=0;
    int out_channel=0,kernel_size=0;

    //Read in parameters and put them in the matrix

    // ERROR IS IN THIS BLOCK
    float w = 0.0;
    while((line=fgets(buffer,bufferSize,fstream))!=NULL){
        if(nnet->layerTypes[layer]==1){
            out_channel = nnet->convLayer[layer][0];
            kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
            if(oc>=out_channel){
                if (param==0)
                {
                    param = 1;
                }
                else
                {
                    param = 0;
                    layer++;
                    if(nnet->layerTypes[layer]==1){
                        out_channel = nnet->convLayer[layer][0];
                        kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
                    }                    
                }
                oc=0;
                ic=0;
                kernel=0;
            }
        }
        else{
            if(i>=nnet->layerSizes[layer+1]){
                if (param==0)
                {
                    param = 1;
                }
                else
                {
                    param = 0;
                    layer++;
                }
                i=0;
                j=0;
            }
        }

        if(nnet->layerTypes[layer]==1){
            if(param==0){
                record = strtok_r(line,",\n", &tmpptr);
                while(record != NULL)
                {   

                    w = (float)atof(record);
                    nnet->conv_matrix[layer][oc][ic][kernel] = w;
                    kernel++;
                    if(kernel==kernel_size){
                        kernel = 0;
                        ic++;
                    }
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr=NULL;
                kernel=0;
                ic=0;
                oc++;
            }
            else{
                record = strtok_r(line,",\n", &tmpptr);
                while(record != NULL)
                {   

                    w = (float)atof(record);
                    nnet->conv_bias[layer][oc] = w;
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr=NULL;
                oc++;
            }
        }
        else{
            record = strtok_r(line,",\n", &tmpptr);
            while(record != NULL)
            {   
                w = (float)atof(record);
                nnet->matrix[layer][param][i][j] = w;                
                j++;
                record = strtok_r(NULL, ",\n", &tmpptr);
            }
            tmpptr=NULL;
            j=0;
            i++;
        }            
    }
    // ERROR IS IN THE BLOCK ABOVE
    printf("load matrix done\n");

    //printf("input size: %d\n",nnet->inputSize);
    struct Matrix input_prev_matrix = {input, 1, nnet->inputSize};
    float o[nnet->outputSize];
    struct Matrix output = {o, nnet->outputSize, 1};
    evaluate_conv(nnet, &input_prev_matrix, &output);
    
    float largest = output.data[0];
    nnet->target = 0;
    for(int o=1;o<nnet->outputSize;o++){
        if(output.data[o]>largest){
            largest = output.data[o];
            nnet->target = o;
        }
    }

    struct Matrix *weights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));

    for(int layer=0;layer<nnet->numLayers;layer++){
        if(nnet->layerTypes[layer]==1) continue;
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data =\
                    (float*)malloc(sizeof(float)*weights[layer].row * weights[layer].col);
        
        int n=0;
        for(int i=0;i<weights[layer].col;i++){
            for(int j=0;j<weights[layer].row;j++){
                weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                n++;
            }
        }
        bias[layer].col = nnet->layerSizes[layer+1];
        bias[layer].row = (float)1;
        bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);
        for(int i=0;i<bias[layer].col;i++){
            bias[layer].data[i] = nnet->matrix[layer][1][i][0];
        }
    } 
    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);
    return nnet;
}


void destroy_conv_network(struct NNet *nnet)
{
    int i=0, row=0;
    if (nnet!=NULL)
    {
        for(i=0; i<nnet->numLayers; i++)
        {
            if(nnet->layerTypes[i]==1) continue;
            for(row=0;row<nnet->layerSizes[i+1];row++)
            {
                //free weight and bias arrays
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }
            //free pointer to weights and biases
            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }
        for(i=0;i<nnet->convLayersNum;i++){
            int in_channel = nnet->convLayer[i][1];
            int out_channel = nnet->convLayer[i][0];
            for(int oc=0;oc<out_channel;oc++){
                for(int ic=0;ic<in_channel;ic++){
                    free(nnet->conv_matrix[i][oc][ic]);
                }
                free(nnet->conv_matrix[i][oc]);
            }
            free(nnet->conv_matrix[i]);
            free(nnet->conv_bias[i]);
        }
        free(nnet->conv_bias);
        free(nnet->conv_matrix);
        for(i=0;i<nnet->convLayersNum;i++){
            free(nnet->convLayer[i]);
        }
        free(nnet->convLayer);
        free(nnet->weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->layerTypes);
        free(nnet->matrix);
        free(nnet);
    }
}


void sort(float *array, int num, int *ind){
    float tmp;
    int tmp_ind;
    for(int i = 0; i < num; i++){
        //printf("%d, %d, %d\n", i, ind[i], array[ind[i]]);
        array[i] = array[ind[i]];
    }
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < (num - i - 1); j++)
        {
            if (array[j] < array[j + 1])
            {
                tmp = array[j];
                tmp_ind = ind[j];
                array[j] = array[j + 1];
                ind[j] = ind[j+1];
                array[j + 1] = tmp;
                ind[j+1] = tmp_ind;
            }
        }
    }
}


void sort_layers(int numLayers, int *layerSizes, int wrong_node_length, int *wrong_nodes_map){
	int wrong_nodes_tmp[wrong_node_length];
	memset(wrong_nodes_tmp, 0, sizeof(int)*wrong_node_length);
	int j = 0;
	int count_node = 0;
	for (int layer = 1; layer < numLayers; layer++){
		count_node += layerSizes[layer]; 
        // printf("%d, %d\n", count_node, count_node-layerSizes[layer]);
		for (int i = 0; i < wrong_node_length; i++){
			if(wrong_nodes_map[i]<count_node && wrong_nodes_map[i]>=count_node-layerSizes[layer]){
				wrong_nodes_tmp[j] = wrong_nodes_map[i];
                //printf("%d, %d;", j, wrong_nodes_tmp[j]);
				j++;
			}
		}
		
	}
    memcpy(wrong_nodes_map, wrong_nodes_tmp, sizeof(int)*wrong_node_length);
    // printf("%d\n", wrong_nodes_map[0]);
}


void set_input_constraints(struct Interval *input,
                        lprec *lp, int *rule_num, int inputSize)
{
    int Ncol = inputSize;
    REAL row[1];
    int colno[1];
    set_add_rowmode(lp, TRUE);
    for(int var=1;var<Ncol+1;var++){
        colno[0] = var;
        row[0] = 1;
        if (input->upper_matrix.data[var-1] == input->lower_matrix.data[var-1]) {
            add_constraintex(lp, 1, row, colno, EQ,\
                        input->upper_matrix.data[var-1]);
            *rule_num += 1;
        } else {
            add_constraintex(lp, 1, row, colno, LE,\
                        input->upper_matrix.data[var-1]);
            add_constraintex(lp, 1, row, colno, GE,\
                        input->lower_matrix.data[var-1]);
            *rule_num += 2;
        }
    }
    set_add_rowmode(lp, FALSE);
}


void set_node_constraints(lprec *lp, float *equation,
                        int start, int *rule_num,
                        int sig, int inputSize)
{
    int Ncol = inputSize;
    REAL row[Ncol+1];
    set_add_rowmode(lp, TRUE);
    row[0] = 0;
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start+j-1];
    }
    if(sig==1){
        add_constraintex(lp, 1, row, NULL, GE, -equation[inputSize+start]);
    }
    else{
        add_constraintex(lp, 1, row, NULL, LE, -equation[inputSize+start]);
    }
    *rule_num += 1;
    set_add_rowmode(lp, FALSE);
}


float set_output_constraints(lprec *lp, float *equation,
                int start_place, int *rule_num, int inputSize,
                int is_max, float *output, float *input_prev)
{
    int Ncol = inputSize;
    REAL row[Ncol+1];
    set_add_rowmode(lp, TRUE);
    row[0] = 0;
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start_place+j-1];
    }
    if(is_max){
        add_constraintex(lp, 1, row, NULL, GE,\
                    -equation[inputSize+start_place]);
        set_maxim(lp);
    }
    else{
        add_constraintex(lp, 1, row, NULL, LE,\
                    -equation[inputSize+start_place]);
        set_minim(lp);
    }
    *rule_num += 1;
    
    set_add_rowmode(lp, FALSE);
    
    set_obj_fnex(lp, Ncol+1, row, NULL);
    //write_lp(lp, "model3.lp");
    int ret = 0;

    //printf("in1\n");

    set_timeout(lp, 30);
    ret = solve(lp);

    //printf("in2,%d\n",ret);
    
    int feasible = 0;
    if(ret == OPTIMAL || ret == SUBOPTIMAL){
        int Ncol = inputSize;
        double row[Ncol+1];
        *output = get_objective(lp) + equation[inputSize+start_place];
        get_variables(lp, row);
        for(int j=0;j<Ncol;j++){
            input_prev[j] = (float)row[j];
        }
        feasible = 1;
    }
    else if(ret == TIMEOUT) {
        feasible = -1;
        printf("Timout while solving LP \n");
    }
    else if(ret == INFEASIBLE) {
        feasible = 0;
    }
    else {
        printf("Abort - Unexpected LP solver return value: %d \n", ret);
        exit(1);
    }
    
    del_constraint(lp, *rule_num);
    *rule_num -= 1; 
    
    return feasible;
}

float set_wrong_node_constraints(lprec *lp,
                float *equation, int start, int *rule_num,
                int inputSize, int is_max, float *output)
{

    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    row[0] = 0;
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start+j-1];
    }
    if(is_max){
        set_maxim(lp);
    }
    else{
        set_minim(lp);
    }
    
    set_obj_fnex(lp, Ncol+1, row, NULL);
    int ret = solve(lp);
    if(ret == OPTIMAL){
        int Ncol = inputSize;
        REAL row[Ncol+1];
        get_variables(lp, row);
        *output = get_objective(lp)+equation[inputSize+start];
    }
    else{
        //printf("unsat!\n");
        unsat = 1;
    }
    return unsat;
}

int find_in_h(int needle, int* h, int h_size) {
    for (int i=0; i<h_size; i++){
        if (needle == h[i]) {
            return 1;
        }
    }
    return 0;
}

void initialize_input_interval(struct NNet* nnet,
                float *input, int input_size,
                int *h, int h_size,
                float *u, float *l, float *u_bounds, float *l_bounds)
{
    for(int i =0;i<input_size;i++){
        // make the entries in the hitting set constant
        if (find_in_h(i,h,h_size)) {
            // make lower and upper bounds the same, this will affect how the constraints
            // are added later on
            float ERR = 1e-3;
            u[i] = input[i]+ERR;
            l[i] = input[i]-ERR;
            if(u[i] > nnet->max) {
                u[i] = nnet->max;
            }
            if(l[i] < nnet->min) {
                l[i] = nnet->min;
            }
        } else {
            u[i] = u_bounds[i];
            if(u[i] > nnet->max) {
                u[i] = nnet->max;
            }
            l[i] = l_bounds[i];
            if(l[i] < nnet->min) {
                l[i] = nnet->min;
            }
        }
    }
    // used for biases
    u[input_size] = 1;
    l[input_size] = 1;

}


void load_inputs(int img, int inputSize, float *input){
    FILE *fstream = fopen("text_inputs/0_spellbinding.csv","r");
    if (fstream == NULL)
    {
        exit(1);
    }
    int bufferSize = 10240*5;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (int i = 0; i<inputSize; i++)
    {
        input[i] = atof(record);
        record = strtok(NULL,",\n");
    }
    free(buffer);
    fclose(fstream);

}


void denormalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i=0; i<nnet->inputSize;i++)
    {
        input->data[i] = input->data[i] * (nnet->max - nnet->min) + nnet->min;
    }

    /*
     * You might want to customize the denormalization function as well
     */
}


void denormalize_input_interval(struct NNet *nnet, struct Interval *input){
    denormalize_input(nnet, &input->upper_matrix);
    denormalize_input(nnet, &input->lower_matrix);
}


void normalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = (input->data[i] - nnet->min) / (nnet->max - nnet->min);
    }

    /*
     * You might want to customize the normalization function as well
     */
}


void normalize_input_interval(struct NNet *nnet, struct Interval *input){
    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);
}


void forward_prop_conv(struct NNet *network,
            struct Matrix *input, struct Matrix *output){
    evaluate_conv(network, input, output);
    float t = output->data[network->target];
    for(int o=0;o<network->outputSize;o++){
        output->data[o] -= t; 
    }
}


int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output){
    int layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;

    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];
    struct Matrix Z = {z, 1, inputSize};
    struct Matrix A = {a, 1, inputSize};

    memcpy(Z.data, input->data, nnet->inputSize*sizeof(float));

    for(layer=0;layer<numLayers;layer++){
        A.row = nnet->bias[layer].row;
        A.col = nnet->bias[layer].col;
        memcpy(A.data, nnet->bias[layer].data, A.row*A.col*sizeof(float));

        matmul_with_bias(&Z, &nnet->weights[layer], &A);
        if(layer<numLayers-1){
            relu(&A);
        }
        memcpy(Z.data, A.data, A.row*A.col*sizeof(float));
        Z.row = A.row;
        Z.col = A.col;
        
    }

    memcpy(output->data, A.data, A.row*A.col*sizeof(float));
    output->row = A.row;
    output->col = A.col;

    return 1;
}



int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output){
    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int outputSize   = nnet->outputSize;

    float ****matrix = nnet->matrix;
    float ****conv_matrix = nnet->conv_matrix;

    float tempVal;
    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];

    
    //printf("start evaluate\n");
    for (i=0; i < nnet->inputSize; i++)
    {
        z[i] = input->data[i];

    }
    //printf("nnet input size: %d\n",nnet->inputSize);

    int out_channel=0, in_channel=0, kernel_size=0;
    int stride=0, padding=0;

    for (layer = 0; layer<(numLayers); layer++)
    {

        memset(a, 0, sizeof(float)*nnet->maxLayerSize);

        //printf("layer %d: %d\n",layer, nnet->layerTypes[layer]);        
        if(nnet->layerTypes[layer]==0){
            for (i=0; i < nnet->layerSizes[layer+1]; i++){
                float **weights = matrix[layer][0];
                float **biases  = matrix[layer][1];
                tempVal = 0.0;

                //Perform weighted summation of inputs
                for (j=0; j<nnet->layerSizes[layer]; j++){
                    tempVal += z[j]*weights[i][j];
                }

                //Add bias to weighted sum
                tempVal += biases[i][0];

                //Perform ReLU
                if (tempVal<0.0 && layer<(numLayers-1)){
                     //printf( "doing RELU on layer %u\n", layer );
                    tempVal = 0.0;
                }
                a[i]=tempVal;
            }
            for(j=0;j<nnet->maxLayerSize;j++){
                //if(layer==2 && j<100) printf("%d %f\n",j, a[j]);
                //if(layer==5 && j<100) printf("%d %f\n",j, a[j]);
                z[j] = a[j];
            }
        }
        else{
            out_channel = nnet->convLayer[layer][0];
            //printf("out_channel %d\n",out_channel);
            in_channel = nnet->convLayer[layer][1];
            //printf("in_channel %d\n",in_channel);
            kernel_size = nnet->convLayer[layer][2];
            //printf("kernel_size %d\n",kernel_size);
            stride = nnet->convLayer[layer][3];
            //printf("stride %d\n",stride);
            padding = nnet->convLayer[layer][4];
            //printf("padding %d\n",padding);
            //size is the input size in each channel
            int size = sqrt(nnet->layerSizes[layer]/in_channel);
            //printf("size %d\n",size);
            //padding size is the input size after padding
            int padding_size = size+2*padding;
            //this is only for compressed model
            if(kernel_size%2==1){
                padding_size += 1;
            }
            //printf("size with padding %d\n",padding_size);
            //out_size is the output size in each channel after kernel
            int out_size = 0;

            /*
             * If you find your padding strategies are different from the one implemented.
             * You might want to change it here.
             */
            float tmp_out_size =  (padding_size-(kernel_size-1)-1)/stride+1;
            if(tmp_out_size == (int)tmp_out_size){
                out_size = (int)tmp_out_size;
            }
            else{
                out_size = (int)(tmp_out_size)-1;
            }
            //printf("out size %d\n",out_size);

            float *z_new = (float*)malloc(sizeof(float)*padding_size*padding_size*in_channel);

            // this part adds in any padding to z_new
            memset(z_new, 0, sizeof(float)*padding_size*padding_size*in_channel);
            for(int ic=0;ic<in_channel;ic++){
                for(int h=0;h<size;h++){
                    for(int w=0;w<size;w++){

                        z_new[ic*padding_size*padding_size+padding_size*(h+padding)+w+padding] =\
                                                            z[ic*size*size+size*h+w];
                    }
                }
            }

            // this part does the convolution
            for(int oc=0;oc<out_channel;oc++){
                for(int oh=0;oh<out_size;oh++){
                    for(int ow=0;ow<out_size;ow++){
                        int start = ow*stride+oh*stride*padding_size;
                        for(int kh=0;kh<kernel_size;kh++){
                            for(int kw=0;kw<kernel_size;kw++){
                                for(int ic=0;ic<in_channel;ic++){
                                    a[oc*out_size*out_size+oh*out_size+ow] +=\
                                    conv_matrix[layer][oc][ic][kh*kernel_size+kw]*\
                                    z_new[ic*padding_size*padding_size+padding_size*kh+kw+start];
                                }
                            }
                        }
                        a[oc*out_size*out_size+ow+oh*out_size]+=nnet->conv_bias[layer][oc];
                    }
                }
            }
            for(j=0;j<nnet->maxLayerSize;j++){
                
                if(a[j]<0){
                    a[j] = 0;
                }
                z[j] = a[j];
                //printf("%f ",a[j]);
            }
            
            //printf("\n");
            free(z_new);
        }
    }

    for (i=0; i<outputSize; i++){
        output->data[i] = a[i];
    }
    
    return 1;
}


void backward_prop_conv(struct NNet *nnet, float *grad,
                     int R[][nnet->maxLayerSize]){
    int i, j, layer;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    float grad_upper[maxLayerSize];
    float grad_lower[maxLayerSize];
    float grad1_upper[maxLayerSize];
    float grad1_lower[maxLayerSize];
    memcpy(grad_upper, nnet->matrix[numLayers-1][0][nnet->target],\
             sizeof(float)*nnet->layerSizes[numLayers-1]);
    memcpy(grad_lower, nnet->matrix[numLayers-1][0][nnet->target],\
             sizeof(float)*nnet->layerSizes[numLayers-1]);
    int start_node = 0;
    for(int l=1; l<nnet->numLayers-1;l++){
        start_node += nnet->layerSizes[l];
    }
    for(int i=0;i<nnet->layerSizes[nnet->numLayers-1];i++){
        //printf("%d, %f %f\n", start_node+i, grad_upper[i], grad_lower[i]);
        grad[start_node+i] = (grad_upper[i]>-grad_lower[i])?grad_upper[i]:-grad_lower[i];
        //printf("%d, %f\n", start_node+i, grad[start_node+i]);
    }

    for(layer = numLayers-2;layer>-1;layer--){
        //printf("layer:%d , %d\n", layer, nnet->layerTypes[layer]);
        float **weights;
        memset(grad1_upper, 0, sizeof(float)*maxLayerSize);
        memset(grad1_lower, 0, sizeof(float)*maxLayerSize);

        if(nnet->layerTypes[layer]!=1){
            weights = nnet->matrix[layer][0];
            if(layer != 0){
                for(j=0;j<nnet->layerSizes[numLayers-1];j++){
                    if(R[layer][j]==0){
                        grad_upper[j] = grad_lower[j] = 0;
                    }
                    else if(R[layer][j]==1){
                        grad_upper[j] = (grad_upper[j]>0)?grad_upper[j]:0;
                        grad_lower[j] = (grad_lower[j]<0)?grad_lower[j]:0;
                    }

                    for(i=0;i<nnet->layerSizes[numLayers-1];i++){
                        if(weights[j][i]>=0){
                            grad1_upper[i] += weights[j][i]*grad_upper[j]; 
                            grad1_lower[i] += weights[j][i]*grad_lower[j]; 
                        }
                        else{
                            grad1_upper[i] += weights[j][i]*grad_lower[j]; 
                            grad1_lower[i] += weights[j][i]*grad_upper[j]; 
                        }
                    }
                }
            }
            else{
                for(j=0;j<nnet->layerSizes[numLayers-1];j++){
                    if(R[layer][j]==0){
                        grad_upper[j] = grad_lower[j] = 0;
                    }
                    else if(R[layer][j]==1){
                        grad_upper[j] = (grad_upper[j]>0)?grad_upper[j]:0;
                        grad_lower[j] = (grad_lower[j]<0)?grad_lower[j]:0;
                    }

                    for(i=0;i<inputSize;i++){
                        if(weights[j][i]>=0){
                            grad1_upper[i] += weights[j][i]*grad_upper[j]; 
                            grad1_lower[i] += weights[j][i]*grad_lower[j]; 
                        }
                        else{
                            grad1_upper[i] += weights[j][i]*grad_lower[j]; 
                            grad1_lower[i] += weights[j][i]*grad_upper[j]; 
                        }
                    }
                }
            }
        }
        else{
            break;
        }

        
        if(layer!=0 && nnet->layerTypes[layer-1]!=1){
            //printf("%d, %d\n", layer, nnet->layerSizes[layer]);
            memcpy(grad_upper,grad1_upper,sizeof(float)*nnet->layerSizes[numLayers-1]);
            memcpy(grad_lower,grad1_lower,sizeof(float)*nnet->layerSizes[numLayers-1]);
            int start_node = 0;
            for(int l=1; l<layer;l++){
                start_node += nnet->layerSizes[l];
            }
            for(int i=0;i<nnet->layerSizes[layer];i++){
                grad[start_node+i] = (grad_upper[i]>-grad_lower[i])?grad_upper[i]:-grad_lower[i];
                //printf("%d, %f %f\n", start_node+i, grad_upper[i], grad_lower[i]);
                //printf("%d, %f\n", start_node+i, grad[start_node+i]);
            }
        }
        else{
            break;
            //memcpy(grad->lower_matrix.data, grad1_lower, sizeof(float)*inputSize);
            //memcpy(grad->upper_matrix.data, grad1_upper, sizeof(float)*inputSize);
        }
    }
}


void sym_fc_layer(struct SymInterval *sInterval,
                    struct SymInterval *new_sInterval, struct NNet *nnet,
                    int layer, int err_row) {
    
    struct Matrix weights = nnet->weights[layer];
    struct Matrix bias = nnet->bias[layer];

    matmul(sInterval->eq_matrix, &weights, new_sInterval->eq_matrix);
    if(err_row>0){
        (*sInterval->err_matrix).row = ERR_NODE;
        matmul(sInterval->err_matrix, &weights, new_sInterval->err_matrix);
        (*new_sInterval->err_matrix).row = (*sInterval->err_matrix).row = err_row;
    }

    int inputSize = nnet->inputSize;
    for (int i=0; i < nnet->layerSizes[layer+1]; i++){
        (*new_sInterval->eq_matrix).data[inputSize+i*(inputSize+1)] += \
            bias.data[i];
    }

    (*sInterval->err_matrix).col = (*new_sInterval->err_matrix).col =\
                        nnet->layerSizes[layer+1];
}


void sym_conv_layer(struct SymInterval *sInterval,
                    struct SymInterval *new_sInterval,
                    struct NNet *nnet, int layer, int err_row) {
    // start handling conv layers
    int inputSize = nnet->inputSize;
    (*new_sInterval->eq_matrix).row = inputSize+1;
    (*new_sInterval->eq_matrix).col = nnet->layerSizes[layer+1];
    (*sInterval->err_matrix).row = (*new_sInterval->err_matrix).row = err_row;
    (*sInterval->err_matrix).col =\
            (*new_sInterval->err_matrix).col =\
            nnet->layerSizes[layer+1];
    //layer is conv
    int out_channel = nnet->convLayer[layer][0];
    int in_channel = nnet->convLayer[layer][1];
    int kernel_size = nnet->convLayer[layer][2];
    int stride = nnet->convLayer[layer][3];
    int padding = nnet->convLayer[layer][4];
    //size is the input size in each channel
    int size = sqrt(nnet->layerSizes[layer]/in_channel);
    //padding size is the input size after padding
    int padding_size = size+2*padding;
    //out_size is the output size in each channel after kernel 
    //int out_size = (int)((padding_size-(kernel_size-1))/stride+1);
    //printf("%d:%d %d %d\n", layer, size, padding_size, out_size);

    //this is only for compressed model
    if(kernel_size%2==1){
        padding_size += 1;
    }
    //out_size is the output size in each channel after kernel
    int out_size = 0;

    float tmp_out_size =  (padding_size-(kernel_size-1)-1)/stride+1;
    if(tmp_out_size == (int)tmp_out_size){
        out_size = (int)tmp_out_size;
    }
    else{
        out_size = (int)(tmp_out_size)-1;
    }

    float *new_new_equation = (float*)malloc(sizeof(float)*\
                    padding_size*padding_size*in_channel*(inputSize+1));
    memset(new_new_equation, 0, sizeof(float)*\
                    padding_size*padding_size*in_channel*(inputSize+1));
    float *new_new_equation_err = (float*)malloc(sizeof(float)*\
                    padding_size*padding_size*in_channel*ERR_NODE);
    memset(new_new_equation_err, 0, sizeof(float)*\
                padding_size*padding_size*in_channel*ERR_NODE);

    for(int ic=0;ic<in_channel;ic++){
        for(int h=0;h<size;h++){
            for(int w=0;w<size;w++){
                for(int k=0;k<inputSize+1;k++){

                    int loc_nn = (ic*padding_size*padding_size+\
                            padding_size*(h+padding)+w+padding)*(inputSize+1)+k;
                    int loc_eq = (ic*size*size+size*h+w)*(inputSize+1)+k;
                    new_new_equation[loc_nn] = (*sInterval->eq_matrix).data[loc_eq];

                }
                for(int k=0;k<err_row;k++){

                    int loc_nn = (ic*padding_size*padding_size+\
                            padding_size*(h+padding)+w+padding)*(ERR_NODE)+k;
                    int loc_eq = (ic*size*size+size*h+w)*(ERR_NODE)+k;
                    new_new_equation_err[loc_nn] = (*sInterval->err_matrix).data[loc_eq];

                }
            }
        }
    }
    

    for(int oc=0;oc<out_channel;oc++){
        for(int oh=0;oh<out_size;oh++){
            for(int ow=0;ow<out_size;ow++){
                if(err_row>0){

                    int start = ow*stride+oh*stride*padding_size;
                    for(int k=0;k<inputSize+1;k++){
                        for(int kh=0;kh<kernel_size;kh++){
                            for(int kw=0;kw<kernel_size;kw++){
                                for(int ic=0;ic<in_channel;ic++){

                                    int loc_eq = (oc*out_size*out_size+\
                                            oh*out_size+ow)*(inputSize+1)+k;
                                    int loc_nn = (ic*padding_size*padding_size+\
                                            padding_size*kh+kw+start)*(inputSize+1)+k;
                                    (*new_sInterval->eq_matrix).data[loc_eq] +=\
                                            nnet->conv_matrix[layer][oc][ic][kh*kernel_size+kw]*\
                                            new_new_equation[loc_nn];
                                    
                                }
                            }
                        }
                        if(k==inputSize){
                            (*new_sInterval->eq_matrix).data[(oc*out_size*out_size+ow+oh*out_size)*(inputSize+1)+k]+=nnet->conv_bias[layer][oc];
                        }
                    }
                    for(int k=0;k<err_row;k++){
                        for(int kh=0;kh<kernel_size;kh++){
                            for(int kw=0;kw<kernel_size;kw++){
                                for(int ic=0;ic<in_channel;ic++){

                                    int loc_er = (oc*out_size*out_size+oh*\
                                            out_size+ow)*ERR_NODE+k;
                                    int loc_nn = (ic*padding_size*padding_size+\
                                            padding_size*kh+kw+start)*ERR_NODE+k;
                                    (*new_sInterval->err_matrix).data[loc_er] +=\
                                            nnet->conv_matrix[layer][oc][ic][kh*kernel_size+kw]*\
                                            new_new_equation_err[loc_nn];

                                }
                            }
                        }
                    }
                }
                else{

                    int start = ow*stride+oh*stride*padding_size;
                    for(int k=0;k<inputSize+1;k++){
                        for(int kh=0;kh<kernel_size;kh++){
                            for(int kw=0;kw<kernel_size;kw++){
                                for(int ic=0;ic<in_channel;ic++){
                                    (*new_sInterval->eq_matrix).data[(oc*out_size*out_size+oh*out_size+ow)*(inputSize+1)+k] += nnet->conv_matrix[layer][oc][ic][kh*kernel_size+kw]*\
                                            new_new_equation[(ic*padding_size*padding_size+padding_size*kh+kw+start)*(inputSize+1)+k];
                                    
                                }
                            }
                        }
                        if(k==inputSize){

                            int loc_eq = (oc*out_size*out_size+ow+oh*out_size)*\
                                        (inputSize+1)+k;
                            (*new_sInterval->eq_matrix).data[loc_eq]+=\
                                        nnet->conv_bias[layer][oc];

                        }
                    }
                }
            }
        }
    }
}


// calculate the upper and lower bound for the ith node in each layer
void relu_bound(struct SymInterval *sInterval,
                struct NNet *nnet, 
                struct Interval *input, int i, int layer, int err_row, 
                float *low, float *up){
    float tempVal_upper=0.0, tempVal_lower=0.0;
    int inputSize = nnet->inputSize;

    float needed_outward_round = 0;
    if(NEED_OUTWARD_ROUND) {
        needed_outward_round = OUTWARD_ROUND;
    }
    
    for(int k=0;k<inputSize+1;k++){
        float weight = (*sInterval->eq_matrix).data[k+i*(inputSize+1)];
        if(weight>=0){
            tempVal_lower +=\
                    weight * input->lower_matrix.data[k]-needed_outward_round;
            tempVal_upper +=\
                    weight * input->upper_matrix.data[k]+needed_outward_round;
        }
        else{
            tempVal_lower +=\
                    weight * input->upper_matrix.data[k]-needed_outward_round;
            tempVal_upper +=\
                    weight * input->lower_matrix.data[k]+needed_outward_round;       
        } 
    }

    if(err_row>0){
        
        for(int err_ind=0;err_ind<err_row;err_ind++){
            float error_value = (*sInterval->err_matrix).data[err_ind+i*ERR_NODE];
            if(error_value > 0){
                tempVal_upper += error_value;
            }
            else{
                tempVal_lower += error_value;
            }
        }
    }

    *up = tempVal_upper;
    *low = tempVal_lower;
}

int relax_relu(struct NNet *nnet, struct SymInterval *sym_interval,
    float lower_bound, float upper_bound, int i,
    int err_row, int *wrong_node_length, int *wcnt) {
    int inputSize = nnet->inputSize;

    //printf("relu relaxation\n");
    if (upper_bound<=0.0){
        upper_bound = 0.0;
        for(int k=0;k<inputSize+1;k++){
            (*sym_interval->eq_matrix).data[k+i*(inputSize+1)] = 0;
        }
        for(int err_ind=0;err_ind<err_row;err_ind++){
            (*sym_interval->err_matrix).data[err_ind+i*ERR_NODE] = 0;
        }
        return 0;
    }
    else if(lower_bound>=0.0){
        return 2;
    }
    else{
        //wrong node length includes the wrong nodes in convolutional layers
        *wrong_node_length += 1;
        *wcnt += 1;
        //printf("wrong: %d,%d:%f, %f\n",layer, i, lower_bound, upper_bound);
        
        for(int k=0;k<inputSize+1;k++){
            (*sym_interval->eq_matrix).data[k+i*(inputSize+1)] *=\
                            upper_bound / (upper_bound - lower_bound);
        }
        for(int err_ind=0;err_ind<err_row;err_ind++){
            (*sym_interval->err_matrix).data[err_ind+i*ERR_NODE] *=\
                        upper_bound / (upper_bound - lower_bound);
        }
        
        (*sym_interval->err_matrix).data[*wrong_node_length-1+i*ERR_NODE] -=\
                                            upper_bound*lower_bound/\
                                            (upper_bound-lower_bound);
        return 1;
    }
}


// relax the relu layers and get the new symbolic equations
int sym_relu_layer(struct SymInterval *new_sInterval,
                    struct Interval *input,
                    struct Interval *output,
                    struct NNet *nnet, 
                    int R[][nnet->maxLayerSize],
                    int layer, int err_row,
                    int *wrong_nodes_map, 
                    int*wrong_node_length,
                    int *node_cnt)
{    
    //record the number of wrong nodes
    int wcnt = 0;

    for (int i=0; i < nnet->layerSizes[layer+1]; i++)
    {
        float tempVal_upper=0.0, tempVal_lower=0.0;
        relu_bound(new_sInterval, nnet, input, i, layer, err_row,\
                    &tempVal_lower, &tempVal_upper);
        
        //Perform ReLU relaxation
        if(layer == nnet->numLayers - 1) {
            output->upper_matrix.data[i] = tempVal_upper;
            output->lower_matrix.data[i] = tempVal_lower;
        }
        else {
            R[layer][i] = relax_relu(nnet, new_sInterval, tempVal_lower,
                tempVal_upper, i, err_row, wrong_node_length, &wcnt);

            if(R[layer][i] == 1) {
                wrong_nodes_map[(*wrong_node_length) - 1] = *node_cnt;
            }
        }
        (*node_cnt) += 1;  
    }

    return wcnt;
}


void forward_prop_interval_equation_linear_conv(struct NNet *nnet,
                            struct Interval *input,
                             struct Interval *output, float *grad,
                             int *wrong_nodes_map, int *wrong_node_length,
                             int *full_wrong_node_length,
                             float *equation_conv, float *equation_conv_err,
                             int *err_row_conv)
{
    int node_cnt=0;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int maxLayerSize   = nnet->maxLayerSize;
    
    ERR_NODE = 5000;
    float *equation_err = (float*)malloc(sizeof(float) *\
                            ERR_NODE*maxLayerSize);
    memset(equation_err, 0, sizeof(float)*ERR_NODE*maxLayerSize);
    float *new_equation_err = (float*)malloc(sizeof(float) *\
                            ERR_NODE*maxLayerSize);
    memset(new_equation_err, 0, sizeof(float)*ERR_NODE*maxLayerSize);

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(int)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer
    float *equation = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    memset(equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
    float *new_equation = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    memset(new_equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);


    struct Matrix equation_matrix = {
                (float*)equation, inputSize+1, inputSize
            };
    struct Matrix new_equation_matrix = {
                (float*)new_equation, inputSize+1, inputSize
            };

    // The real row number for index is ERR_NODE, 
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
    for (int layer = 0; layer<numLayers; layer++)
    {
        // printf("%d, %d", layer, nnet->layerSizes[layer]);
        memset(new_equation, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_err,0,sizeof(float)*ERR_NODE*maxLayerSize);
        
        if(nnet->layerTypes[layer]==0) {
            // FC layer
            
            sym_fc_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
            
            /*store equation and error matrix for later splitting*/
            if(layer == 0 || (CHECK_ADV_MODE && nnet->layerTypes[layer]==0 && \
                            nnet->layerTypes[layer-1]==1)) {

                memcpy(equation_conv, new_equation,\
                        sizeof(float)*(inputSize+1)*maxLayerSize);
                memcpy(equation_conv_err, new_equation_err,\
                        sizeof(float)*ERR_NODE*maxLayerSize);
                *err_row_conv = err_row;

            }
            
            int wcnt = sym_relu_layer(&new_sInterval, input, output, nnet, R,
                                layer, err_row, wrong_nodes_map,
                                wrong_node_length, &node_cnt);
            
            *full_wrong_node_length = *full_wrong_node_length + wcnt;

        }
        else{

            sym_conv_layer(&sInterval, &new_sInterval, nnet, layer, err_row);

            if(layer == 0){

                memcpy(equation_conv, new_equation,\
                        sizeof(float)*(inputSize+1)*maxLayerSize);
                memcpy(equation_conv_err, new_equation_err,\
                        sizeof(float)*ERR_NODE*maxLayerSize);
                *err_row_conv = err_row;

            }

            sym_relu_layer(&new_sInterval, input, output, nnet, R, layer,
                err_row, wrong_nodes_map, wrong_node_length, &node_cnt);
        }
        //printf("\n");
        memcpy(equation, new_equation, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_err, new_equation_err, sizeof(float)*(ERR_NODE)*maxLayerSize);
        equation_matrix.col = new_equation_matrix.col;
        equation_err_matrix.col = new_equation_err_matrix.col;
        err_row = *wrong_node_length;
    }

    backward_prop_conv(nnet, grad, R);

    free(equation_err);
    free(new_equation_err);
    free(equation);
    free(new_equation);
}
