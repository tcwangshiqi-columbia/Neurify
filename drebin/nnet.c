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


int PROPERTY = 5;
char *LOG_FILE = "logs/log.txt";
int INF = 1;
int ACCURATE_MODE=1;

struct timeval start,finish,last_finish;
//FILE *fp;


int search_inputs(float *input, int input_length, int node_cnt){
    int wrong_ind=-1;
    for(int wn=0;wn<input_length;wn++){
        if(node_cnt == (int)input[wn]){
            wrong_ind = wn;
        }
    }
    return wrong_ind;
}

//Take in a .nnet filename with path and load the network from the file
//Inputs:  filename - const char* that specifies the name and path of file
//Outputs: void *   - points to the loaded neural network
struct NNet *load_network(const char* filename, struct Matrix *input_prev_matrix)
{
    int inputSize = input_prev_matrix->col;
    //Load file and check if it exists
    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL)
    {
        printf("Wrong network!\n");
        exit(1);
    }
    //Initialize variables
    int bufferSize = 210906900;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;

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

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->numLayers)+1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }

    if(PROPERTY>=100){
        nnet->inputSize = inputSize;
        nnet->layerSizes[0] = inputSize;
        nnet->maxLayerSize = inputSize;
    }

    //Load the symmetric paramter
    nnet->symmetric = 0;
    //Load Min and Max values of inputs
    nnet->min = MIN_PIXEL;
    nnet->max = MAX_PIXEL;
    //Load Mean and Range of inputs
    nnet->mean = MIN_PIXEL;
    nnet->range = MAX_PIXEL;

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = (float****)malloc(sizeof(float *)*nnet->numLayers);
    
    for (layer = 0; layer<(nnet->numLayers); layer++)
    {
        nnet->matrix[layer] = (float***)malloc(sizeof(float *)*2);
        nnet->matrix[layer][0] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        nnet->matrix[layer][1] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        for (row = 0; row<nnet->layerSizes[layer+1]; row++)
        {
            nnet->matrix[layer][0][row] = (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
            nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
        }
    }

    layer = 0;
    param = 0;
    i=0;
    j=0;
    char *tmpptr=NULL;

    //Read in parameters and put them in the matrix
    float w = 0.0;
    while((line=fgets(buffer,bufferSize,fstream))!=NULL)
    {
        if(i>=nnet->layerSizes[layer+1])
        {
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
        record = strtok_r(line,",\n", &tmpptr);
        int k = 0;
        while(record != NULL)
        {   
            w = (float)atof(record);
            if(PROPERTY>=100 && param == 0 && layer == 0){
                k=search_inputs(input_prev_matrix->data, inputSize, j);
                if(k!=-1){
                    nnet->matrix[layer][param][i][k] = w;
                    //printf("%d %f %f\n",i, input_prev_matrix->data[k], nnet->matrix[layer][param][i][k]);
                    //k++;
                }
            }
            else{
                nnet->matrix[layer][param][i][j] = w;
            }
            
            j++;
            record = strtok_r(NULL, ",\n", &tmpptr);
        }
        tmpptr=NULL;
        j=0;
        i++;
    }

    /*get target output*/
    float o[nnet->outputSize];
    struct Matrix output = {o, nnet->outputSize, 1};
    if(PROPERTY>=100){
        float ei[nnet->inputSize];
        struct Matrix input = {ei, 1, nnet->inputSize};
        for(int ind=0;ind<inputSize;ind++){
            if(ind<inputSize-INF){
                ei[ind] = 1;
            }
            else{
                ei[ind] = 0;
            }
            
        }
        //printMatrix(&input);
        evaluate(nnet, &input, &output);
    }
    else{
        evaluate(nnet, input_prev_matrix, &output);
    }
    
    //printMatrix(&output);

    float largest = -100000.0;
    for(int o=0;o<nnet->outputSize;o++){
        if(output.data[o]>largest){
            largest = output.data[o];
            nnet->target = o;
        }
    }

    /*update weights*/
    float orig_weights[nnet->layerSizes[layer]];
    float orig_bias;
    struct Matrix *weights = malloc(nnet->numLayers*sizeof(struct Matrix));
    nnet->neg_weights = malloc(nnet->numLayers*sizeof(struct Matrix));
    nnet->pos_weights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));
    for(int layer=0;layer<nnet->numLayers;layer++){
        nnet->pos_weights[layer].row = nnet->neg_weights[layer].row\
                                     = weights[layer].row = nnet->layerSizes[layer];
        nnet->pos_weights[layer].col = nnet->neg_weights[layer].col\
                                     = weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data =\
                    (float*)malloc(sizeof(float)*weights[layer].row * weights[layer].col);
        nnet->neg_weights[layer].data =\
                    (float*)malloc(sizeof(float)*weights[layer].row * weights[layer].col);
        nnet->pos_weights[layer].data =\
                    (float*)malloc(sizeof(float)*weights[layer].row * weights[layer].col);
        int n=0;
        if(1){
            /*
             * Make the weights of last layer to minus the target one.
             */
            if(layer==nnet->numLayers-1){
                orig_bias = nnet->matrix[layer][1][nnet->target][0];
                memcpy(orig_weights, nnet->matrix[layer][0][nnet->target],\
                                     sizeof(float)*nnet->layerSizes[layer]);
                for(int i=0;i<weights[layer].col;i++){
                    for(int j=0;j<weights[layer].row;j++){
                        weights[layer].data[n] = nnet->matrix[layer][0][i][j]-orig_weights[j];
                        n++;
                    }
                }
                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);
                for(int i=0;i<bias[layer].col;i++){
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0]-orig_bias;
                }
            }
            else{
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
        }
        else{
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
        
        memset(nnet->neg_weights[layer].data, 0,\
                     sizeof(float)*weights[layer].col*weights[layer].row);
        memset(nnet->pos_weights[layer].data, 0,\
                     sizeof(float)*weights[layer].col*weights[layer].row);

        for(i=0;i<weights[layer].row*weights[layer].col;i++){
            if(weights[layer].data[i]>=0){
                nnet->pos_weights[layer].data[i] = weights[layer].data[i];
            }
            else{
                nnet->neg_weights[layer].data[i] = weights[layer].data[i];
            }
        }
        
    } 
    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);
    
    nnet->maxLayerSize = inputSize;
    for(i=0;i<nnet->numLayers;i++){
        if(nnet->layerSizes[i]>nnet->maxLayerSize){
            nnet->maxLayerSize = nnet->layerSizes[i];
        }
    }    


    return nnet;
}


void destroy_network(struct NNet *nnet)
{
    int i=0, row=0;
    if (nnet!=NULL)
    {
        for(i=0; i<(nnet->numLayers); i++)
        {
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
            free(nnet->pos_weights[i].data);
            free(nnet->neg_weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }
        free(nnet->weights);
        free(nnet->neg_weights);
        free(nnet->pos_weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->matrix);
        free(nnet);
    }
}


void calculate_gradient(struct NNet* nnet, float *input, float *u, float *l, struct Interval *grad){

    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    for(int i =0;i<inputSize;i++){
        u[i] = input[i];
        l[i] = input[i];
    }
    struct Matrix input_upper = {u,1,nnet->inputSize};
    struct Matrix input_lower = {l,1,nnet->inputSize};
    struct Interval input_interval = {input_lower, input_upper};
    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {(struct Matrix){o_lower, outputSize, 1},
                                       (struct Matrix){o_upper, outputSize, 1}};
    float *equation_upper = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
    float *equation_lower = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *new_equation_upper = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
    float *new_equation_lower = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);

    forward_prop_interval_equation(nnet, &input_interval, &output_interval,\
                            grad, equation_upper, equation_lower,\
                            new_equation_upper, new_equation_lower);       

    free(equation_upper);
    free(equation_lower);
    free(new_equation_upper);
    free(new_equation_lower);


}

void sort_bottom(float *array, int num, int *ind){
    float tmp;
    int tmp_ind;
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < (num - i - 1); j++)
        {
            if (array[j] > array[j + 1])
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

void sort(float *array, int num, int *ind){
    float tmp;
    int tmp_ind;
    for(int i = 0; i < num; i++){
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

void sort_top(float *array, int num, int *ind){
    float tmp;
    int tmp_ind;
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


void sort_top_gradient(float *grad, int *ind, int inputSize, int indSize){
    float queue[indSize];
    memset(queue, 0, sizeof(float)*indSize);
    float g = 0.0;
    int j = 0;
    for(int i=0;i<inputSize;i++){
        //printf("%d, %f\n",i,grad[i]);
        g = (grad[i]>0)?grad[i]:-grad[i];
        if(i<indSize){
            queue[i] = g;
            ind[i] = i;
            continue;
        }
        if(i==indSize){
            sort_top(queue, indSize, ind);
        }
        if(g>queue[indSize-1]){
            queue[indSize-1] = g;
            ind[indSize-1] = i;
            sort_top(queue, indSize, ind);
        }
    }
}

void sort_bottom_gradient(float *grad, int *ind, int inputSize, int indSize){
    float queue[indSize];
    memset(queue, 0, sizeof(float)*indSize);
    float g = 0.0;
    int j = 0;
    for(int i=0;i<inputSize;i++){
        g = (grad[i]>0)?grad[i]:-grad[i];
        //printf("%d, %f\n",i,grad[i]);
        if(i<indSize){
            queue[i] = g;
            ind[i] = i;
            continue;
        }
        if(i==indSize){
            sort_bottom(queue, indSize, ind);
        }
        if(g<queue[indSize-1]){
            queue[indSize-1] = g;
            ind[indSize-1] = i;
            sort_bottom(queue, indSize, ind);
        }
    }
}

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num){
    int Ncol = input->upper_matrix.col;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
    for(int var=1;var<Ncol+1;var++){
        memset(colno, 0, Ncol*sizeof(int));
        /*
        for(int j=1;j<Ncol+1;j++){
            if(var == j){
                row[j]=1;
                colno[0] = var;
            }
            else{
                row[j]=0;
            }
        }
        add_constraintex(lp, 1, row, NULL, LE, input->upper_matrix.data[var-1]);
        add_constraintex(lp, 1, row, NULL, GE, input->lower_matrix.data[var-1]);
        */
        colno[0] = var;
        row[0] = 1;
        add_constraintex(lp, 1, row, colno, LE, input->upper_matrix.data[var-1]);
        add_constraintex(lp, 1, row, colno, GE, input->lower_matrix.data[var-1]);
        *rule_num += 2;
    }
    set_add_rowmode(lp, FALSE);
}

void set_node_constraints(lprec *lp, float *equation, int start, int *rule_num, int sig, int inputSize){
    int Ncol = inputSize;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
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

float set_output_constraints(lprec *lp, float *equation, int start_place, int *rule_num, int inputSize, int is_max, float *output, float *input_prev){
    
    float time_spent=0.0;
    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start_place+j-1];
    }
    if(is_max){
        add_constraintex(lp, 1, row, NULL, GE, -equation[inputSize+start_place]);
        set_maxim(lp);
    }
    else{
        add_constraintex(lp, 1, row, NULL, LE, -equation[inputSize+start_place]);
        set_minim(lp);
    }
    
    set_add_rowmode(lp, FALSE);
    
    set_obj_fnex(lp, Ncol+1, row, NULL);
    *rule_num += 1;
    //write_lp(lp, "model3.lp");
    int ret = 0;

    ret = solve(lp);

    
    if(ret == OPTIMAL){
        int Ncol = inputSize;
        double row[Ncol+1];
        *output = get_objective(lp)+equation[inputSize+start_place];
        get_variables(lp, row);
        for(int j=0;j<Ncol;j++){
            input_prev[j] = (float)row[j];
        }
    }
    else{
        //printf("unsat!\n");
        unsat = 1;
    }
    del_constraint(lp, *rule_num);
    *rule_num -= 1; 
    return unsat;
}

float set_wrong_node_constraints(lprec *lp, float *equation, int start, int *rule_num, int inputSize, int is_max, float *output){
    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    memset(row, 0, Ncol*sizeof(float));

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


void initialize_input_interval(struct NNet* nnet, int img, int inputSize, float *input, float *u, float *l){
    if(PROPERTY == 0){
        for(int i =0;i<inputSize;i++){
            u[i] = input[i]+INF;
            l[i] = input[i]-INF;
        }
    }
    if(PROPERTY == 1){
        int indSize = 200;
        float grad_upper[inputSize], grad_lower[inputSize];
        struct Interval grad_interval = {(struct Matrix){grad_upper, 1, inputSize},
                                        (struct Matrix){grad_lower, 1, inputSize}};
        int ind[indSize];
        memset(ind, 0, sizeof(int)*indSize);

        float tmp_input[inputSize];
        memcpy(tmp_input, input, sizeof(float)*inputSize);
        normalize_input(nnet, &(struct Matrix){tmp_input,1,inputSize});
        calculate_gradient(nnet, tmp_input, u, l, &grad_interval);
        /*
        for(int i=0;i<inputSize;i++){
            printf("%d: %f\n", i, grad_upper[i]);
        }
        */
        memcpy(u, input, sizeof(float)*inputSize);
        memcpy(l, input, sizeof(float)*inputSize);
        sort_bottom_gradient(grad_upper, ind, nnet->inputSize, indSize);
        for(int i=0;i<indSize;i++){
            u[ind[i]] = input[ind[i]]+INF;
            l[ind[i]] = input[ind[i]]-INF;
            //printf("%d: %f\n", ind[i], grad_upper[ind[i]]);
        }
    }
    if(PROPERTY == 2){
        int indSize = 200;
        float grad_upper[inputSize], grad_lower[inputSize];
        struct Interval grad_interval = {(struct Matrix){grad_upper, 1, inputSize},
                                        (struct Matrix){grad_lower, 1, inputSize}};
        int ind[indSize];
        memset(ind, 0, sizeof(int)*indSize);

        float tmp_input[inputSize];
        memcpy(tmp_input, input, sizeof(float)*inputSize);
        normalize_input(nnet, &(struct Matrix){tmp_input,1,inputSize});
        calculate_gradient(nnet, tmp_input, u, l, &grad_interval);
        memcpy(u, input, sizeof(float)*inputSize);
        memcpy(l, input, sizeof(float)*inputSize);
        sort_top_gradient(grad_upper, ind, nnet->inputSize, indSize);
        for(int i=0;i<indSize;i++){
            u[ind[i]] = input[ind[i]]+INF;
            l[ind[i]] = input[ind[i]]-INF;
            //printf("%d: %f\n", ind[i], grad_upper[ind[i]]);
        }
    }
    if(PROPERTY == 3){
        memcpy(u, input, sizeof(float)*inputSize);
        memcpy(l, input, sizeof(float)*inputSize);
        for(int i=0;i<600;i+=3){
            u[i] = input[i]+INF;
            l[i] = input[i]-INF;
            //printf("%d: %f\n", ind[i], grad_upper[ind[i]]);
        }
    }
    if(PROPERTY==101){
        for(int i=0;i<inputSize;i++){
            if(i<inputSize-INF){
                input[i] = 1.0;
                u[i] = 1.0;
                l[i] = 1.0;
            }
            else{
                input[i] = 0.0;
                u[i] = 1.0;
                l[i] = 0.0;
            }
        }
    }
}

int load_inputs_img(int img, float *input){
    int inputSize;
    if(img>=100000){
        printf("images over 100000!\n");
        exit(1);
    }
    char str[12];
    char img_name[18] = "images/image";
    sprintf(str, "%d", img);
    FILE *fstream = fopen(strcat(img_name,str),"r");
    if (fstream == NULL)
    {
        printf("no input:%s!\n", img_name);
        exit(1);
    }
    int bufferSize = 10240;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    inputSize = atoi(record);
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i<inputSize; i++)
    {
        input[i] = atoi(record);
        record = strtok(NULL,",\n");
    }
    free(buffer);
    fclose(fstream);
    return inputSize;
}


int load_inputs_drebin(int app, float *input){
    int inputSize;
    if(app>=100000){
        printf("apps over 100000!\n");
        exit(1);
    }
    char str[12];
    char app_name[18] = "apps_new/app";
    sprintf(str, "%d", app);
    FILE *fstream = fopen(strcat(app_name,str),"r");
    if (fstream == NULL)
    {
        printf("no input:%s!\n", app_name);
        exit(1);
    }
    int bufferSize = 10240;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    inputSize = atoi(record);
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i<inputSize; i++)
    {
        input[i] = atoi(record);
        record = strtok(NULL,",\n");
    }
    for(int i=0;i<INF;i++){
        int r = (int)(((float)rand()/(float)RAND_MAX)*545334); 
        if(search_inputs(input, inputSize+i-1, r) == -1){
            input[i+inputSize] = r;
        }
    }
    inputSize = inputSize+INF;
    free(buffer);
    fclose(fstream);
    return inputSize;
}


int forward_prop_interval(struct NNet *network, struct Interval *input, struct Interval *output){
    int i,j,layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    float z_upper[nnet->maxLayerSize];
    float z_lower[nnet->maxLayerSize];
    float a_upper[nnet->maxLayerSize];
    float a_lower[nnet->maxLayerSize];
    struct Matrix Z_upper = {z_upper, 1, inputSize};
    struct Matrix A_upper = {a_upper, 1, inputSize};
    struct Matrix Z_lower = {z_lower, 1, inputSize};
    struct Matrix A_lower = {a_lower, 1, inputSize};

    for (i=0; i < nnet->inputSize; i++)
    {
        z_upper[i] = input->upper_matrix.data[i];
        z_lower[i] = input->lower_matrix.data[i];
    }

    memcpy(Z_upper.data, input->upper_matrix.data, nnet->inputSize*sizeof(float));
    memcpy(Z_lower.data, input->lower_matrix.data, nnet->inputSize*sizeof(float));

    float temp_upper[maxLayerSize*maxLayerSize];
    float temp_lower[maxLayerSize*maxLayerSize];
    struct Matrix Temp_upper = {temp_upper,maxLayerSize,maxLayerSize};
    struct Matrix Temp_lower = {temp_upper,maxLayerSize,maxLayerSize};

    for(layer=0;layer<numLayers;layer++){
        A_upper.row = A_lower.row = nnet->bias[layer].row;
        A_upper.col = A_lower.col = nnet->bias[layer].col;
        memcpy(A_upper.data, nnet->bias[layer].data, A_upper.row*A_upper.col*sizeof(float));
        memcpy(A_lower.data, nnet->bias[layer].data, A_lower.row*A_lower.col*sizeof(float));

        for(i=0;i<nnet->weights[layer].row*nnet->weights[layer].col;i++){
            if(nnet->weights[layer].data[i]>=0){
                Temp_upper.data[i] = Z_upper.data[i%5];
                Temp_lower.data[i] = Z_lower.data[i%5];
            }
            else{
                Temp_upper.data[i] = Z_lower.data[i%5];
                Temp_lower.data[i] = Z_upper.data[i%5];
            }
            Temp_upper.row = Temp_lower.row = nnet->weights[layer].row;
            Temp_upper.col = Temp_lower.col = Z_upper.col;
        }

        matmul_with_bias(&Temp_upper, &nnet->weights[layer], &A_upper);
        matmul_with_bias(&Temp_lower, &nnet->weights[layer], &A_lower);
        Z_upper.row = Z_lower.row = nnet->bias[layer].row;
        Z_upper.col = Z_lower.col = nnet->bias[layer].col;
        for(i=0;i<nnet->bias[layer].col*nnet->bias[layer].row;i++){
            Z_upper.data[i] = A_upper.data[i*A_upper.col+i];
            Z_lower.data[i] = A_lower.data[i*A_lower.col+i];
        }
        if(layer<numLayers){
            relu(&Z_upper);
            relu(&Z_lower);
        }
    }

    memcpy(output->upper_matrix.data, Z_upper.data, Z_upper.row*Z_upper.col*sizeof(float));
    memcpy(output->lower_matrix.data, Z_lower.data, Z_lower.row*Z_lower.col*sizeof(float));
    output->upper_matrix.row = output->lower_matrix.row = Z_upper.row;
    output->upper_matrix.col = output->lower_matrix.col = Z_upper.col;
    return 1;
}

void denormalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i=0; i<nnet->inputSize;i++)
    {
        input->data[i] = input->data[i]*(nnet->range) + nnet->mean;
    }
}

void denormalize_input_interval(struct NNet *nnet, struct Interval *input){
    denormalize_input(nnet, &input->upper_matrix);
    denormalize_input(nnet, &input->lower_matrix);
}

void normalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i=0; i<nnet->inputSize;i++)
    {
        if (input->data[i]>nnet->max)
        {
            input->data[i] = (nnet->max-nnet->mean)/(nnet->range);
        }
        else if (input->data[i]<nnet->min)
        {
            input->data[i] = (nnet->min-nnet->mean)/(nnet->range);
        }
        else
        {
            input->data[i] = (input->data[i]-nnet->mean)/(nnet->range);
        }
    }
}


void normalize_input_interval(struct NNet *nnet, struct Interval *input){
    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);
}


int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output){
    int i,j,layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

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


int evaluate(struct NNet *network, struct Matrix *input, struct Matrix *output)
{
    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = inputSize;
    for(i=0;i<nnet->numLayers;i++){
        if(nnet->layerSizes[i]>maxLayerSize){
            maxLayerSize = nnet->layerSizes[i];
        }
    }

    float ****matrix = nnet->matrix;

    float tempVal;
    float z[maxLayerSize];
    float a[maxLayerSize];
    memset(z, 0, maxLayerSize*sizeof(float));
    memset(a,0, maxLayerSize*sizeof(float));

    for (i=0; i < nnet->inputSize; i++)
    {
        z[i] = input->data[i];
    }

    for (layer = 0; layer<(numLayers); layer++)
    {
        memset(a, 0, maxLayerSize*sizeof(float));
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            float **weights = matrix[layer][0];
            float **biases  = matrix[layer][1];
            tempVal = 0.0;

            //Perform weighted summation of inputs
            for (j=0; j<nnet->layerSizes[layer]; j++)
            {
                tempVal += z[j]*weights[i][j];
            }
            //Add bias to weighted sum
            tempVal += biases[i][0];
            
            //Perform ReLU
            if (tempVal<0.0 && layer<(numLayers-1))
            {
                // printf( "doing RELU on layer %u\n", layer );
                tempVal = 0.0;
            }
            a[i]=tempVal;
        }
        memcpy(z,a,maxLayerSize*sizeof(float));
    }

    for (i=0; i<outputSize; i++)
    {
        output->data[i] = a[i];
    }
    return 1;
}


int evaluate_interval(struct NNet *network, struct Interval *input, struct Interval *output)
{
    int i,j,layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    float ****matrix = nnet->matrix;

    float tempVal_upper, tempVal_lower;
    float z_upper[nnet->maxLayerSize];
    float z_lower[nnet->maxLayerSize];
    float a_upper[nnet->maxLayerSize];
    float a_lower[nnet->maxLayerSize];

    for (i=0; i < nnet->inputSize; i++)
    {
        z_upper[i] = input->upper_matrix.data[i];
        z_lower[i] = input->lower_matrix.data[i];
    }

    for (layer = 0; layer<(numLayers); layer++)
    {
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            float **weights = matrix[layer][0];
            float **biases  = matrix[layer][1];
            tempVal_upper = tempVal_lower = 0.0;

            for (j=0; j<nnet->layerSizes[layer]; j++)
            {
                if(weights[i][j]>=0){
                    tempVal_upper += z_upper[j]*weights[i][j];
                    tempVal_lower += z_lower[j]*weights[i][j];
                }
                else{
                    tempVal_upper += z_lower[j]*weights[i][j];
                    tempVal_lower += z_upper[j]*weights[i][j];
                }
            }

            //Add bias to weighted sum
            tempVal_lower += biases[i][0];
            tempVal_upper += biases[i][0];

            //Perform ReLU
            if(layer<(numLayers-1)){
                if (tempVal_lower<0.0){
                    tempVal_lower = 0.0;
                }
                if (tempVal_upper<0.0){
                    tempVal_upper = 0.0;
                }
            }
            a_upper[i] = tempVal_upper;
            a_lower[i] = tempVal_lower;
        }
        for(j=0;j<nnet->maxLayerSize;j++){
            z_upper[j] = a_upper[j];
            z_lower[j] = a_lower[j];
        }
    }

    for (i=0; i<outputSize; i++)
    {
        output->upper_matrix.data[i] = a_upper[i];
        output->lower_matrix.data[i] = a_lower[i];
    }

    return 1;
}


int evaluate_interval_equation(struct NNet *network, struct Interval *input, struct Interval *output)
{
    int i,j,k,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    float ****matrix = nnet->matrix;

    // equation is the temp equation for each layer
    //float equation_upper[maxLayerSize][inputSize+1];
    //float equation_lower[maxLayerSize][inputSize+1];
    //float new_equation_upper[maxLayerSize][inputSize+1];
    //float new_equation_lower[maxLayerSize][inputSize+1];

    float *equation_upper = (float*)malloc(sizeof(float)*(inputSize+1)*maxLayerSize);
    float *equation_lower = (float*)malloc(sizeof(float)*(inputSize+1)*maxLayerSize);
    float *new_equation_upper = (float*)malloc(sizeof(float)*(inputSize+1)*maxLayerSize);
    float *new_equation_lower = (float*)malloc(sizeof(float)*(inputSize+1)*maxLayerSize);

    memset(equation_upper, 0, sizeof(float)*maxLayerSize*(inputSize+1));
    memset(equation_lower, 0, sizeof(float)*maxLayerSize*(inputSize+1));
    memset(new_equation_upper, 0, sizeof(float)*maxLayerSize*(inputSize+1));
    memset(new_equation_lower, 0, sizeof(float)*maxLayerSize*(inputSize+1));

    float tempVal_upper, tempVal_lower;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }

    for (layer = 0; layer<(numLayers); layer++)
    {
        memset(new_equation_upper, 0, sizeof(float)*maxLayerSize*(inputSize+1));
        memset(new_equation_lower, 0, sizeof(float)*maxLayerSize*(inputSize+1));
        
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            float **weights = matrix[layer][0];
            float **biases  = matrix[layer][1];
            tempVal_upper = tempVal_lower = 0.0;

            for (j=0; j<nnet->layerSizes[layer]; j++)
            {
                for(k=0;k<inputSize+1;k++){
                    if(weights[i][j]>=0){
                        new_equation_upper[i*(inputSize+1)+k] += equation_upper[j*(inputSize+1)+k]*weights[i][j];
                        new_equation_lower[i*(inputSize+1)+k] += equation_lower[j*(inputSize+1)+k]*weights[i][j];
                    }
                    else{
                        new_equation_upper[i*(inputSize+1)+k] += equation_lower[j*(inputSize+1)+k]*weights[i][j];
                        new_equation_lower[i*(inputSize+1)+k] += equation_upper[j*(inputSize+1)+k]*weights[i][j];
                    }
                }
            }
            for(k=0;k<inputSize;k++){
                if(new_equation_lower[i*(inputSize+1)+k]>=0){
                    tempVal_lower += new_equation_lower[i*(inputSize+1)+k] * input->lower_matrix.data[k];
                }
                else{
                    tempVal_lower += new_equation_lower[i*(inputSize+1)+k] * input->upper_matrix.data[k];
                }
                if(new_equation_upper[i*(inputSize+1)+k]>=0){
                    tempVal_upper += new_equation_upper[i*(inputSize+1)+k] * input->upper_matrix.data[k];
                }
                else{
                    tempVal_upper += new_equation_upper[i*(inputSize+1)+k] * input->lower_matrix.data[k];
                }  
            }
            new_equation_lower[i*(inputSize+1)+inputSize] += biases[i][0];
            new_equation_upper[i*(inputSize+1)+inputSize] += biases[i][0];
            tempVal_lower += new_equation_lower[i*(inputSize+1)+inputSize];
            tempVal_upper += new_equation_upper[i*(inputSize+1)+inputSize];

            //Perform ReLU
            if(layer<(numLayers-1)){
                if (tempVal_lower<0.0){
                    tempVal_lower = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                    new_equation_upper[i*(inputSize+1)+inputSize] = tempVal_upper;
                }
                if (tempVal_upper<0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                }
            }
            else{
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
            }
        }
        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
    }

    free(equation_upper);
    free(equation_lower);
    free(new_equation_upper);
    free(new_equation_lower);

    return 1;
}


int forward_prop_interval_equation(struct NNet *network, struct Interval *input,
                             struct Interval *output, struct Interval *grad,
                             float *equation_upper, float *equation_lower,
                             float *new_equation_upper, float *new_equation_lower)
{
    int i,j,k,layer;
    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer
    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(new_equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(new_equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    struct Interval equation_inteval = {
        (struct Matrix){(float*)equation_lower, inputSize+1, inputSize},
        (struct Matrix){(float*)equation_upper, inputSize+1, inputSize}
    };
    struct Interval new_equation_inteval = {
        (struct Matrix){(float*)new_equation_lower, inputSize+1, inputSize},
        (struct Matrix){(float*)new_equation_upper, inputSize+1, inputSize}
    };                                       

    float tempVal_upper, tempVal_lower;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }

    for (layer = 0; layer<(numLayers); layer++)
    {

        memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);

        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];
        
        matmul(&equation_inteval.upper_matrix, &nnet->pos_weights[layer],
                 &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &nnet->neg_weights[layer],
                 &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &nnet->pos_weights[layer],
                 &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix, &nnet->neg_weights[layer],
                 &new_equation_inteval.lower_matrix);


        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            tempVal_upper = tempVal_lower = 0.0;
            if(NEED_OUTWARD_ROUND){
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k]-OUTWARD_ROUND;
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k]-OUTWARD_ROUND;
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k]+OUTWARD_ROUND;
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k]+OUTWARD_ROUND;
                    }  
                }
            }
            else{
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k];
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k];
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k];
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k];
                    }  
                }
            }
            
            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];
            tempVal_lower += new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper += new_equation_upper[inputSize+i*(inputSize+1)];

            //Perform ReLU
            //printf("%f %f\n", tempVal_lower, tempVal_upper);
            if(layer<(numLayers-1)){
                if (tempVal_upper<=0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                }
                else{
                    R[layer][i] = 1;
                }

                if (tempVal_lower<=0.0){
                    //tempVal_lower = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                    new_equation_upper[inputSize+i*(inputSize+1)] = tempVal_upper;
                }
                else{
                    
                     //0:mean lower<0 upper<0 [0,0]
                     //1:mean lower<0 upper>0 [0,1]
                     //2:mean lower>0 upper>0 [1,1]
                    
                    R[layer][i] = 2;
                }
            }
            else{
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
            }

        }

        //printf("\n");
        memcpy(equation_upper, new_equation_upper,\
                 sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower,\
                 sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row \
                    = equation_inteval.upper_matrix.row \
                    = new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col \
                    = equation_inteval.upper_matrix.col \
                    = new_equation_inteval.lower_matrix.col;
    }
    //backward_prop(nnet, grad, R);
    //printf("%f %f", grad->upper_matrix.data[10], grad->lower_matrix.data[10]);

    return 1;
}


void backward_prop(struct NNet *nnet, float *grad,
                     int R[][nnet->maxLayerSize]){
    int i, j, layer;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = inputSize;
    for(i=0;i<nnet->numLayers;i++){
        if(nnet->layerSizes[i]>maxLayerSize){
            maxLayerSize = nnet->layerSizes[i];
        }
    }

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
        grad[start_node+i] = (grad_upper[i]>-grad_lower[i])?grad_upper[i]:-grad_lower[i];
        //printf("%d, %f\n", start_node+i, grad[start_node+i]);
    }

    for(layer = numLayers-2;layer>-1;layer--){
        float **weights = nnet->matrix[layer][0];
        memset(grad1_upper, 0, sizeof(float)*maxLayerSize);
        memset(grad1_lower, 0, sizeof(float)*maxLayerSize);

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

        
        if(layer!=0){
            //printf("%d, %d\n", layer, nnet->layerSizes[layer]);
            memcpy(grad_upper,grad1_upper,sizeof(float)*nnet->layerSizes[numLayers-1]);
            memcpy(grad_lower,grad1_lower,sizeof(float)*nnet->layerSizes[numLayers-1]);
            int start_node = 0;
            for(int l=1; l<layer;l++){
                start_node += nnet->layerSizes[l];
            }
            for(int i=0;i<nnet->layerSizes[layer];i++){
                grad[start_node+i] = (grad_upper[i]>-grad_lower[i])?grad_upper[i]:-grad_lower[i];
                //printf("%d, %f\n", start_node+i, grad[start_node+i]);
            }
        }
        else{
            //memcpy(grad->lower_matrix.data, grad1_lower, sizeof(float)*inputSize);
            //memcpy(grad->upper_matrix.data, grad1_upper, sizeof(float)*inputSize);
        }
    }
}

int forward_prop_interval_equation2(struct NNet *network, struct Interval *input,
                             struct Interval *output, struct Interval *grad,
                             float *equation_upper, float *equation_lower,
                             float *new_equation_upper, float *new_equation_lower)
{
    int i,j,k,layer;
    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    float upper_s_lower=0.0;

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer
    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(new_equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(new_equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    struct Interval equation_inteval = {
        (struct Matrix){(float*)equation_lower, inputSize+1, inputSize},
        (struct Matrix){(float*)equation_upper, inputSize+1, inputSize}
    };
    struct Interval new_equation_inteval = {
        (struct Matrix){(float*)new_equation_lower, inputSize+1, inputSize},
        (struct Matrix){(float*)new_equation_upper, inputSize+1, inputSize}
    };                                       

    float tempVal_upper, tempVal_lower;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }

    for (layer = 0; layer<(numLayers); layer++)
    {
        for(i=0;i<maxLayerSize;i++){
            memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
            memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        }

        matmul(&equation_inteval.upper_matrix, &nnet->pos_weights[layer],\
                 &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &nnet->neg_weights[layer],\
                 &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &nnet->pos_weights[layer],\
                 &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix,\
                         &nnet->neg_weights[layer],\
                         &new_equation_inteval.lower_matrix);
        
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            tempVal_upper = tempVal_lower = 0.0;

            if(NEED_OUTWARD_ROUND){
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k]-OUTWARD_ROUND;
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k]-OUTWARD_ROUND;
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k]+OUTWARD_ROUND;
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k]+OUTWARD_ROUND;
                    }  
                }
            }
            else{
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k];
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k];
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->upper_matrix.data[k];
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] *\
                         input->lower_matrix.data[k];
                    }  
                }
            }
            
            new_equation_lower[inputSize+i*(inputSize+1)] += nnet->bias[layer].data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += nnet->bias[layer].data[i];
            tempVal_lower += new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper += new_equation_upper[inputSize+i*(inputSize+1)];
            upper_s_lower += new_equation_upper[inputSize+i*(inputSize+1)];

            //Perform ReLU
            if(layer<(numLayers-1)){
                if (tempVal_upper<=0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                    R[layer][i] = 0;
                }
                else if(tempVal_lower>=0.0){
                    R[layer][i] = 2;
                }
                else{
                    tempVal_lower = 0.0;
                    //printf("wrong node: ");
                    if(upper_s_lower>=0.0){
                        //printf("upper's lower >= 0\n");
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                        }
                    }
                    else{
                        //printf("upper's lower <= 0\n");
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                            new_equation_upper[k+i*(inputSize+1)] = 0;
                        }
                        new_equation_upper[inputSize+i*(inputSize+1)] = tempVal_upper;
                        //new_equation_upper[inputSize+i*(inputSize+1)] -= upper_s_lower;
                    }
                    R[layer][i] = 1;
                }
            }
            else{
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
            }
        }
        //printf("\n");
        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =  new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =  new_equation_inteval.lower_matrix.col;
    }

    //backward_prop(nnet, grad, R);

    return 1;
}


int forward_prop_interval_equation_linear(struct NNet *network, struct Interval *input,
                                     struct Interval *output, struct Interval *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower)
{
    int i,j,k,layer;

    struct NNet* nnet = network;
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
    float upper_s_lower=0.0;
    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }

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
            tempVal_upper = tempVal_lower = 0.0;

            if(NEED_OUTWARD_ROUND){
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                    }  
                }
            }
            else{
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    }  
                }
            }
            
            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];
            tempVal_lower += new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper += new_equation_upper[inputSize+i*(inputSize+1)];
            upper_s_lower += new_equation_upper[inputSize+i*(inputSize+1)];

            //Perform ReLU
            if(layer<(numLayers-1)){
                if (tempVal_upper<=0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                    R[layer][i] = 0;
                }
                else if(tempVal_lower>=0.0){
                    R[layer][i] = 2;
                }
                else{
                    tempVal_lower = 0.0;
                    //printf("wrong node: ");
                    if(upper_s_lower>=0.0){
                        //printf("upper's lower >= 0\n");
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                        }
                    }
                    else{
                        //printf("upper's lower <= 0\n");
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                            new_equation_upper[k+i*(inputSize+1)] =\
                                                    new_equation_upper[k+i*(inputSize+1)]*\
                                                    tempVal_upper / (tempVal_upper-upper_s_lower);
                        }
                        //new_equation_upper[inputSize+i*(inputSize+1)] = tempVal_upper;
                        new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*upper_s_lower/\
                                                            (tempVal_upper-upper_s_lower);
                    }
                    R[layer][i] = 1;
                }
            }
            else{
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
            }
        }
        //printf("\n");
        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
    }

    //backward_prop(nnet, grad, R);

    return 1;
}


int forward_prop_interval_equation_linear2(struct NNet *network, struct Interval *input,
                                     struct Interval *output, float *grad,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower,
                                     int *wrong_nodes, int *wrong_node_length,
                                     float *wrong_up_s_up, float *wrong_up_s_low,
                                     float *wrong_low_s_up, float *wrong_low_s_low)
{
    int i,j,k,layer;
    int node_cnt=0;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = inputSize;
    for(i=0;i<nnet->numLayers;i++){
        if(nnet->layerSizes[i]>maxLayerSize){
            maxLayerSize = nnet->layerSizes[i];
        }
    }

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
    
    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }    

    for (layer = 0; layer<(numLayers); layer++)
    {
        //printf("0:%f %f \n",equation_lower[0],equation_upper[0] );
        if(layer!=0) memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        if(layer!=0) memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        //printf("1:%f %f \n",equation_lower[0],equation_upper[0] );
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
            tempVal_upper = tempVal_lower = 0.0;
            lower_s_upper = upper_s_lower = 0.0;

            if(NEED_OUTWARD_ROUND){
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                        lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                        lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                    }  
                }
            }
            else{
                for(k=0;k<inputSize;k++){
                    if(layer==0){
                        if(new_equation_lower[k+i*(inputSize+1)]!=new_equation_upper[k+i*(inputSize+1)]){
                            printf("wrong!\n");
                        }
                    }
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                        lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                        lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }  
                }
            }
            
            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];
            tempVal_lower += new_equation_lower[inputSize+i*(inputSize+1)];
            lower_s_upper += new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper += new_equation_upper[inputSize+i*(inputSize+1)];
            upper_s_lower += new_equation_upper[inputSize+i*(inputSize+1)];

            //Perform ReLU
            //printf("%f %f\n", tempVal_lower, tempVal_upper);
            if(layer<(numLayers-1)){
                wrong_up_s_up[node_cnt] = tempVal_upper;
                wrong_up_s_low[node_cnt] = upper_s_lower;
                wrong_low_s_up[node_cnt] = lower_s_upper;
                wrong_low_s_low[node_cnt] = tempVal_lower;
                if (tempVal_upper<=0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                    R[layer][i] = 0;
                }
                else if(tempVal_lower>=0.0){
                    R[layer][i] = 2;
                }
                else{

                    wrong_nodes[*wrong_node_length] = node_cnt;

                    *wrong_node_length += 1;
                    if(lower_s_upper>0 || upper_s_lower<0){
                        //printf("%d,%d:%f, %f, %f, %f\n",layer, i, tempVal_lower, lower_s_upper, upper_s_lower, tempVal_upper );
                    }
                    //printf("wrong node: ");
                    if(upper_s_lower<0.0){
                        for(k=0;k<inputSize+1;k++){
                            new_equation_upper[k+i*(inputSize+1)] =\
                                                    new_equation_upper[k+i*(inputSize+1)]*\
                                                    tempVal_upper / (tempVal_upper-upper_s_lower);
                        }
                        new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*upper_s_lower/\
                                                            (tempVal_upper-upper_s_lower);
                    }

                    if(lower_s_upper<0.0){
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                        }
                    }
                    else{
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] =\
                                                    new_equation_lower[k+i*(inputSize+1)]*\
                                                    lower_s_upper / (lower_s_upper- tempVal_lower);
                        }
                    }
                    R[layer][i] = 1;
                }
            }
            else{
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
            }
            node_cnt++;
        }
        

        //printf("\n");
        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
    }
    backward_prop(nnet, grad, R);
    return 1;
}



