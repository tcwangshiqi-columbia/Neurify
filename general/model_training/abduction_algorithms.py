"""
Implementation of Algorithms 1 and 2 of the paper "Abduction-Based Explanation for ML Models", AAAI-2019
---
Take a model saved with Keras+Tensorflow and save the frozen graph in a .pb format.model.layers[idx].get_config()
Execute this in the '/home/emalfa/Desktop/__Oxford/Marabou/maraboupy' with the command
 %run ./examples/file.py
---
@Author: Emanuele
"""
import itertools
import multiprocessing as mp
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow import keras
from pysat.examples.hitman import Hitman
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear as keras_linear
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import sys
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore

def logger(msg, 
           verbose=False, log_type=''):
    """
    Print stuff nicely.
    """
    if verbose is False:
        pass
    else:
        print("[logger{}]: {}".format('-'+log_type if log_type!='' else '', msg))

def idx2word(indices, window_size, input_len):
    """
    When a feature is selected by MinimumHS, the entire window is selected
    """
    res = []
    for idx in indices:
        for r in range(0, input_len, window_size):
            if idx<r+window_size and idx>=r and (idx not in res):
                res.extend([el for el in range(r, min(r+window_size, input_len))])
    return res

def freeze_session(session, 
                   keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def MinimumHS(subsets, 
              htype='smallest', **kwargs):
    """
    Return a list that contains the minimum-hitting-set of the input, i.e. given a list of lists
     it finds the minimum set that intersects all the others and whose dimension is minimal
    Example: subsets = [[1], [], [2,1], [33]], returns [1,33] (it ignores empty sets)
    Uses pysat Hitman, but includes the possibility to extract the smallest, the largest and a random
     minimum hitting set.
    Input:
        subsets:list of lists
            all the "sets" used to find the minimum-hitting set. Can't contain empty set []
        hype:string
            (optional) 'smallest' resturn the smallest and largest minimal HS, "random-smallest" choses one of the 
             minimal hitting sets at random, 'random' returns one at random everytime is invoked. 
             'MLC' (Minumum Linear Cost) selects the explanation that minimizes the linear distance among all 
             the features (when sorted). Finally, 'LIME' guarantees that the hitting set is both minimal and
             has as much feature in common with the kwargs `lime_set` as possible
        **kwargs:any
            (optional) `lime_set` is a list of features (from a LIME explanation) that is used to search the 
             MinimumHS of minimal size and with as much feature in common with `lime_set` as possible.
    Output:
        hs:list
            minimum hitting set
    """
    if len(subsets) == 0:
        hs = []
        return hs
    if [] in subsets:
        raise Exception("Subsets variable can't contain [] set (empty)")
    h = Hitman(bootstrap_with=subsets, htype='sorted')
    if htype == 'smallest':
        hs = h.get()
        return hs
    elif htype == 'largest':
        for el in h.enumerate():
            hs = el
        return hs
    elif htype == 'random':
        HS = []
        for el in h.enumerate():
            HS += [el]
        hs = random.choice(HS)
        return hs
    elif htype == "random-smallest":
        HS = [h.get()]
        min_size = len(HS[0])
        for el in h.enumerate():
            if len(el) == min_size:
                HS += [el]
        hs = random.choice(HS)
        return hs
    elif htype == 'MLC':
        best_hs, min_cost = [], np.inf
        for el in h.enumerate():
            c = 0
            el.sort()  # sort and calculate linear cost
            for i,_ in enumerate(el[:-1]):
                c += el[i+1] - el[i]
            if c < min_cost:
                best_hs = el
                min_cost = c
        return best_hs
    elif htype == 'LIME' and len(kwargs['lime_set']) > 0:
        lime_set = kwargs['lime_set']
        hs = h.get()
        min_size = len(hs)
        features_in_common = len(set(lime_set).intersection(hs))
        for el in h.enumerate():
            if len(el) == min_size and len(set(lime_set).intersection(el)) > features_in_common:
                hs = el
                features_in_common = len(set(lime_set).intersection(el))
        return hs
    elif htype == "exclude" and len(kwargs['exclude_set']) > 0:
        exclude_set = kwargs['exclude_set']
        hs = h.get()
        min_size = len(hs)
        features_in_common = len(set(exclude_set).intersection(hs))
        for el in h.enumerate():
            if len(el) == min_size and len(set(exclude_set).intersection(el)) < features_in_common:
                hs = el
                features_in_common = len(set(exclude_set).intersection(el))
        return hs
    else:
        raise NotImplementedError("{} is not a vald htype method".format(htype))

def Entails(h, network, input_, input_bounds, output_constraints, 
            verbose=False):
    """
    Implementation of Entails function from "Abduction-Based Explanation for ML Models", AAAI-2019 that
     is solved with Marabou c++ solver.
    Input:
        h:list
            contains the indices of variables that are fixed (i.e., C - h are free vars).
        network:maraboupy.MarabouNetworkTF.MarabouNetworkTF
            marabou network specification
        input_bounds:list
            list of [min, max] lists, one entry for each variable
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        verbose:boolean
            (optional) verbosity of the logger function
    Output:
        vals:dictionary
            dictionary of values {idx: value} where each index refers to the i-th input variable, while the
             value is indeed the value of a successfull adversarial attack if found. 
             If nothing has been found, vals is equal to {}, hence its length is 0.
    """
    # Ranges for Free params ('f_ranges') and Constants ('c_ranges')
    c_ranges = [n for n in h]
    f_ranges = [n for n in range(0, len(input_)) if n not in c_ranges]
    logger("Free Vars(s) {}".format(f_ranges), verbose, "DEBUG")
    logger("Constant Var(s) {}".format(c_ranges), verbose, "DEBUG")
    # Get the input and output variable numbers
    inputVars = network.inputVars[0][0].flatten()
    outputVars = network.outputVars[0].flatten()
    # Set input bounds
    for n in f_ranges:
        network.setLowerBound(inputVars[n], input_bounds[n][0])
        network.setUpperBound(inputVars[n], input_bounds[n][1])    
    # Constants (before and after Free Params) are set equal to their original value
    for n in c_ranges:
        #sets var n = constant
        equation = MarabouUtils.Equation()
        equation.addAddend(1, n)
        equation.setScalar(input_[n])
        network.addEquation(equation)
    # y1>=(y0-\eps) inequality constraint
    #  (which becomes strict with a small epsilon added to one of the outputs)
    # API: \sum_i vars_i*coeffs_i <= scalar
    # 
    y_targ_idx, y_opp_idx, eps = output_constraints
    # eps should be negative, if its positve make it negative
    if eps > 0:
        eps = -eps
    network.addInequality([outputVars[y_targ_idx], outputVars[y_opp_idx]], [1,-1], eps)

    # Call to C++ Marabou solver
    logger("Results for value {}".format(h), verbose, "DEBUG")
    opts = Marabou.createOptions(verbosity=2)
    vals, _ = network.solve(options=opts,verbose=True)
    sys.exit(0)
    return vals

def PickFalseLits(C_setminus_h, filename, input_, input_bounds, output_constraints, 
                  window_size=1, randomize=False, verbose=False):
    """
    Search for a subset of free variables in C_setminus_h s.t. the adversarial attack is still effective.
    Input:
        C_setminus_h:list
            contains the indices of variables that are free.
             Naming is terrible but at least we are consistent with the abduction paper notation
        filename:string
            path to the graph in the format maraboupy.MarabouNetworkTF.MarabouNetworkTF
        input_:list
            flatten list of inputs of the adversarial attack
        input_bounds:list
            list of [min, max] lists, one entry for each variable
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        window_size:int
            length of the window used to encode a single variable
        randomize:boolean
            (optional) free variables in C_setminus_h are shuffled before being processed
        verbose:boolean
            (optional) verbosity of the logger function
    Output:
        C_prime:list
            subset of the original free variables that successfully brought an adversarial attack
    """
    C_prime, adv = [], []  # free variables that we collect, hopefully reduced wrt C_setminus_h
    input_len = len(input_)
    iterator_ = list(range(0, input_len, window_size))
    fixed_vars = [i for i in iterator_ if i not in C_setminus_h]  # vars that are fixed when input_ is found and the function is called
    free_vars = [i for i in iterator_ if i in C_setminus_h]  # vars that are fixed when input_ is found and the function is called
    # shuffle free variables 
    if randomize is True:
        random.shuffle(free_vars)
    for i in free_vars:
        h = fixed_vars + idx2word([i], window_size, input_len)  # fix i-th window
        network = Marabou.read_tf(filename)  # re-initialize the network
        res = Entails(h, network, input_, input_bounds, output_constraints)
        if len(res) != 0:  # if there is an attack
            fixed_vars += idx2word([i], window_size, input_len)  # var will be discarded (the attack is still adversarial)
    adv = [i for i in free_vars if i not in fixed_vars]  # keep all free vars that were really adversarial
    for i in adv:
        C_prime += idx2word([i], window_size, input_len)  # expand window
    logger("{}".format(C_prime), verbose, "DEBUG")
    return C_prime

def smallest_explanation(model, filename, numpy_input, eps, y_hat, output_constraints, window_size, 
                         adv_attacks=False, adv_args=(None, None), sims=10, randomize_pickfalselits=False, HS_maxlen=100, verbose=True):
    """
    Smallest Explanation API, algorithm 2 paper "Abduction-Based Explanation for ML Models", AAAI-2019.
    Input:
        model:tensorflow.keras.models
            model used to make predictions and extract the Marabou graph
        filename:string
            path to the graph in the format maraboupy.MarabouNetworkTF.MarabouNetworkTF
        numpy_input:numpy.array
            input of the adversarial attack in numpy format
        eps:float
            value used to determine bounds, in terms of [min, max] for each input
        y_hat:int
            integer that specifies the output class
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        window_size:int
            length of the window used to encode a single variable
        adv_attacks:boolean
            (optional) exploit PickFalseLiterals with adversarial attack routines where number of attacks is minimized
        adv_args:tuple
            (optional) tuple (func, args) used to launch the advresarial attacks routine
        sims:int
            (optional) number of simulations (per number of variables) in the PGD adversarial attacks routine
        randomize_pickfalselits:boolean
            (optional) PickFalseLits function uses a randomized approach to refine the explanations
        HS_maxlen:int
            (optional) max size of GAMMA. Keep it under 150-200 even for machines with lot of memory
        verbose:boolean
            (optional) verbosity of the logger function.
    Output:
        smallest_expl:list
            variables in the minimal explanation
        exec_time:float
            execution time in seconds from the beginning of the execution of the routine
        GAMMA:list
            list of hitting sets
    """
    start_time = time.time()
    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)
    input_bounds = [[input_[i]-eps if input_[i]-eps >= 0 else 0, input_[i]+eps if input_[i]+eps <= 1 else 1] for i in range(input_len)]
    #print(input_bounds)
    #sys.exit(0)
    # input bounds match with Neurify
    GAMMA = []
    timer = 0 
    while True:
        # Keep length of Gamma fixed + keep only smallest elements
        if len(GAMMA) > HS_maxlen:
            GAMMA.sort(key=len)
            GAMMA = GAMMA[:HS_maxlen]
        if GAMMA != []:
            logger("Calculating HS on set of average len = {}, (size of GAMMA={})".format(sum(map(len, GAMMA))/float(len(GAMMA)), len(GAMMA)), verbose, "DEBUG")
        # Generate Minimum Hitting Set
        #h = MinimumHS(GAMMA, htype='LIME', lime_set=[30,31,32,33,34,45,46,47,48,49])
        #h = MinimumHS(GAMMA, htype='exclude', exclude_set=[30,31,32,33,34,45,46,47,48,49])  # Alg. 2 line 2, initially empty
        h = MinimumHS(GAMMA, htype='smallest')
        h = idx2word(h, window_size, input_len)  # fixed vars
        logger("MinimumHS {}".format(h), verbose, "DEBUG")
        # Start procedure
        network = Marabou.read_tf(filename,modelType='savedModel_v2',savedModelTags=['serving_default'])
        # 1. Look for an adversarial attack *before* Entails (at line 5 Alg. 2)
        if adv_attacks is True:
            adv_args = (adv_args[0], adv_args[1][:6] + (h,) + adv_args[1][7:])  # set mask on inputs that should not be tested
            found, res = adv_args[0](*adv_args[1])
            if found is False:
                logger("Adversarial attack not found on free vars {}. Run Entails on vars {}".format([f for f in range(input_len) if f not in h], h), verbose, 'DEBUG')
                res = Entails(h, network, input_, input_bounds, output_constraints, verbose)
            else:
                logger("Adversarial attack found on free vars {}".format([f for f in range(input_len) if f not in h]), verbose, 'DEBUG')
        else:
            logger("Run Entails on vars {}".format(h), verbose, "DEBUG")
            res = Entails(h, network, input_, input_bounds, output_constraints, verbose)  # Algorithm, line 5  
        if len(res) == 0:  # return if there is no attack with h as a smallest explanation
            break
        else:
            logger("Attack found", verbose, "DEBUG")
            C_setminus_h = [c for c in range(input_len) if c not in h]  # free vars used to find an attack, naming consistent with paper notation
            pop_attacks = []
            # 2.1 Search sparse aversarial attacks with sparseRS routine
            if adv_attacks is True:
                mask = [m for m in range(input_len) if m in h]  # vars excluded from the adversarial attacks (i.e., h)
                # Run sparsePGD before trying Entails (which is usually way slower)
                _, _, pop_attacks = adv.optimize_sparseRS(model, numpy_input, input_bounds, y_hat, 
                                                          num_classes=2, k=min(10*window_size, len(C_setminus_h)), sims=sims, mask=mask, PGDargs=(False, eps, 1))
                if len(pop_attacks) != 0:
                    logger("SparseRS is successful with a total of {} attacks".format(len(pop_attacks)), verbose, '')
                    for pop in pop_attacks:
                        candidate_p = np.argwhere(numpy_input.flatten()!=pop.flatten()).flatten().tolist()
                        if candidate_p not in GAMMA:  # add *all* the attacks that have been found
                            GAMMA += [idx2word(candidate_p, window_size, input_len)]
            # 2.2 Refine the attack found with heuristic or run PickFalseLits
            if len(pop_attacks) != 0:  # SparseRS has found at least an attack
                C_setminus_h = [c for c in range(input_len) if c in pop_attacks[-1]]  # set PickFalseLits arg to the smallest attack found with SparseRS
            logger("Run PickFalseLits with pop_attacks size = {}, adv_attacks = {}".format(len(pop_attacks), adv_attacks), verbose, "DEBUG")
            
            #C_prime = PickFalseLits(C_setminus_h, filename, input_, input_bounds, output_constraints, window_size, randomize_pickfalselits, verbose)            
            C_prime = C_setminus_h
            
            if len(C_prime) > 0 and C_prime not in GAMMA:
                GAMMA += [idx2word(C_prime, window_size, input_len)]
        logger("Iteration n.{}, size of GAMMA = {}".format(timer, len(GAMMA)), verbose, "DEBUG")
        timer += 1
    exec_time = time.time() - start_time
    return h, exec_time, GAMMA
