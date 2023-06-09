
import numpy as np
import pandas as pd
import networkx as nx
import os, time, random, argparse, pickle, math, copy
import pprint  #NOTE remove

import algs.utils as utils
import algs.utils_misc as utils_misc
import algs.kernel as kernel


def get_prerun_data(prerun_files, kern_fn):
    # if graph is already kernelized
    partial_kerndat=None
    if kern_fn in prerun_files:
        with open(kern_fn, 'rb') as infile: 
            partial_kerndat = pickle.load(infile)
    return partial_kerndat


def run_kernel(G, A, cliq_verts, k_input, kernel_v2_enabled, ordering_strategy):     
    '''
    
    '''
    print('* Running Kernel, kinp=', k_input)
    
    # reduction rules    
    start = time.time()
    A_kernel, removal_vertices, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A, k_input, kernel_v2_enabled, ordering_strategy)
    end = time.time()   
    time_kernel = end - start
    
    # did the kernel pass or fail?
    passed_kernel=True 
    if len(removal_vertices)>0:
        if removal_vertices[0]==-1:
            passed_kernel=False
        else: 
            # update the clique vertices 
            for v in removal_vertices:
                for cliq in cliq_verts:
                    if v in cliq:
                        cliq.remove(v)
    
    winf = None
    G.remove_nodes_from(removal_vertices)
    if G.order()>0:
        winf =  np.linalg.norm(nx.to_numpy_matrix(G), np.inf)
    
    kernel_dat = {'passed_kernel' : passed_kernel,
                  'A_kernel' : A_kernel,
                  'removal_vertices' : removal_vertices,
                  'num_total_blocks': num_total_blocks,
                  'num_reduced_blocks': num_reduced_blocks,
                  'clique_vertices' : cliq_verts,
                  'n' : G.order(),
                  'm' : G.number_of_edges(),
                  'max_eweight' : utils_misc.get_max_edgeweight(G),
                  'w_inf' : winf,
                  'kinput' : k_input,
                  'kernel_time' : time_kernel}
    
    return kernel_dat
        

def run_origcompare_exp(fname, partial_kerndat, out, k_distinct, k_orig_max):
    '''
    determines if the original comparision experiment should run on current file
    run on kdistinct<=11, witer=0, small/med weight scales
    
    if prerun, updates corresponding out field
    '''
    run_total=True
    
    # graph has already been ran 
    if partial_kerndat != None:
        if partial_kerndat['true_total_kernel'] != None:
            run_total=False
            out['true_total_kernel'] = partial_kerndat['true_total_kernel']
    
    if 'tf' in fname:
        w_iter = utils_misc.get_fname_value(fname, 'witer')
        if w_iter != 0:
            run_total=False
        
        ## run on small/med weight scales
        #w_scale = utils_misc.get_fname_value(fname, 'maxcweight') 
        #if w_scale!=1 and w_scale!=4:
            #run_total=False
        
    #elif 'lv' in fname:
        ## run on small/med weight scales
        #w_scale = utils_misc.get_fname_value(fname, 'scalefac') 
        #if w_scale!=1 and w_scale!=2:
            #run_total=False
            
    # run on small/med weight scales
    w_scale = utils_misc.get_fname_value(fname, 'scalefac') 
    if w_scale!=1 and w_scale!=2:
        run_total=False
    
    if k_distinct>k_orig_max:
        run_total=False
    
    return run_total


def run_distinct_exp(fname, partial_kerndat, out, w_iter_max):
    '''
    determine whether to run algs on current graph w. kdistinct value
    '''
    run_dist=True
    if partial_kerndat != None:
        if partial_kerndat['true_distinct_kernel'] != None:
            run_dist=False
            out['true_distinct_kernel'] = partial_kerndat['true_distinct_kernel']
        
    if 'tf' in fname:
        w_iter = utils_misc.get_fname_value(fname, 'witer')
        if w_iter > w_iter_max:  # NOTE [0,5] weight iterations
            run_dist=False
            
    return run_dist


def run_guess_exp(fname, partial_kerndat, allguesses, kg, k_distinct, k_guess_max):
    '''
    determine whether or not to run the guessing experiment
    '''
    run_guess=True
    if partial_kerndat != None:
        if partial_kerndat['guess_kernels'][kg] != None:
            run_guess=False
            allguesses[kg] = partial_kerndat['guess_kernels'][kg]
    
    if 'tf' in fname:
        w_iter = utils_misc.get_fname_value(fname, 'witer')
        if w_iter != 0:
            run_guess=False
    
    if k_distinct<5:
        run_guess=False
    
    if k_distinct>k_guess_max:
        run_guess=False
    
    return run_guess
    
    
def run(dirname,
        out_dirname,
        first_seed,
        last_seed,
        target_kdistinct,
        k_orig_max,
        k_distinct_max,
        k_guess_max,
        w_iter_max,
        kernel_v2_enabled,
        ordering_strategy):
    '''
    
    '''
    print('in dirname: ', dirname)
    print('out_dirname: ', out_dirname)
    
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname, exist_ok=True)
    
    # collect graph files
    allfiles = utils_misc.get_files(dirname, '.txt')
    files = utils_misc.filter_files_seed(allfiles, first_seed, last_seed)
    
    # already processed graphs
    prerun_files = utils_misc.get_files(out_dirname, '.pkl') 
    count=0
    
    for fname in files: 
        print('\n__________________________________ {}/{} : {}'.format(count, 
                                                                       len(files),
                                                                       fname))
        count+=1
        
        # get post_preprocessing pkl file info
        pklfname = fname[0:-4]+'.pkl'
        with open(pklfname, 'rb') as infile: 
            output = pickle.load(infile)
            
        # collect + write out data
        out = { 'true_total_kernel'     : None, 
                'true_distinct_kernel'  : None,
                'guess_kernels'         : None}
        
        kern_fn = out_dirname+fname.split('/')[-1][0:-4]+'.pkl'
        partial_kerndat = get_prerun_data(prerun_files, kern_fn)      
        
        k_total = output['post_preprocessing']['ktotal']
        k_distinct = output['post_preprocessing']['kdistinct']
        print('ktotal ', k_total, ' kdistinct', k_distinct)
        
        if target_kdistinct is None and k_distinct > k_distinct_max:
            print('skipping for now because kdistinct>kdistinct_max')
            continue
        if target_kdistinct is not None and k_distinct != target_kdistinct:
            print('skipping because kdistinct != target_kdistinct')
            continue

        if output['post_preprocessing']['n']>0:
            G = nx.read_weighted_edgelist(fname, nodetype=int)
        
            #-------------------- 1. run on true_total
            run_total = run_origcompare_exp(fname, partial_kerndat, out,
                                            k_distinct, k_orig_max)
            if run_total:
                print('--------- running true total kernel')
                G_total = G.copy()
                A_total = utils_misc.get_wildcard_adjacency(G_total)
                cvs = output['post_preprocessing']['clique_vertices']
                cliq_verts = copy.deepcopy(cvs)
                
                true_out = run_kernel(G_total, A_total, cliq_verts, k_total, kernel_v2_enabled, ordering_strategy)
                out['true_total_kernel'] = true_out
            
            
            #-------------------- 2. run on true_distinct
            run_dist = run_distinct_exp(fname, partial_kerndat, out, w_iter_max)
            if run_dist:
                print('--------- running true distinct kernel')
                G_distinct = G.copy()
                A_distinct = utils_misc.get_wildcard_adjacency(G_distinct)
                cvs = output['post_preprocessing']['clique_vertices']
                cliq_verts = copy.deepcopy(cvs)
                
                distinct_out = run_kernel(G_distinct, A_distinct, 
                                          cliq_verts, k_distinct, kernel_v2_enabled, ordering_strategy)
                out['true_distinct_kernel'] = distinct_out
            
            
            #-------------------- 3. guessing kinput value
            # -60%, -40%, -20%, +20%, +40%, +60% of true kdistinct value
            allguesses = {-0.6 : None, 
                          -0.4 : None,
                          -0.2 : None, 
                           0.2 : None, 
                           0.4 : None,
                           0.6 : None }
            
            for kg in [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]:
                k_guess = k_distinct - math.floor(k_distinct * kg)
                            
                run_guess = run_guess_exp(fname, partial_kerndat, allguesses, 
                                          kg, k_distinct, k_guess_max)
                # if the kdistinct value is specified, always enable guesses
                if target_kdistinct is not None or run_guess:
                    print('--------- running guess kernel')
                    G_guess = G.copy()
                    A_guess = utils_misc.get_wildcard_adjacency(G_guess)
                    cvs = output['post_preprocessing']['clique_vertices']
                    cliq_verts = copy.deepcopy(cvs)
                    
                    allguesses[kg] = run_kernel(G_guess, A_guess, 
                                                cliq_verts, k_guess, kernel_v2_enabled, ordering_strategy)
                out['guess_kernels'] = allguesses
        
        with open(kern_fn, 'wb') as f: 
            pickle.dump(out, f)


def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-f', '--first_seed', type=int,
        help="enter first seed value in seed range", required=True)
    parser.add_argument('-l', '--last_seed', type=int,
        help="enter last seed value in seed range", required=True)
    parser.add_argument('--kdistinct', type=int,
        help='limit experiments to the instances with the specific kdistinct value', required=False)
    parser.add_argument('--v2', action='store_true',
        help='use an improved kernelization technique based on the nonuniform Fisher inequality (default: False)')
    parser.add_argument('--ordering', choices=['arbitrary', 'keep_first', 'keep_last', 'push_front', 'push_back'],
        help='vertex reordering strategy (default: push_front)', default = 'push_front')
    args = vars(parser.parse_args())
    
    first_seed = args.get('first_seed')
    last_seed = args.get('last_seed')
    target_kdistinct = args.get('kdistinct')
    
    # max parameter values used in ACDA21 experiments
    k_orig_max=11
    k_distinct_max=11
    k_guess_max=7
    w_iter_max=5
    kernel_v2_enabled = args['v2']
    ordering_strategy = {
        'arbitrary': kernel.OrderingStrategy.ARBITRARY,
        'keep_first': kernel.OrderingStrategy.KEEP_FIRST,
        'keep_last': kernel.OrderingStrategy.KEEP_LAST,
        'push_front': kernel.OrderingStrategy.PUSH_FRONT,
        'push_back': kernel.OrderingStrategy.PUSH_BACK,
    }[args['ordering']]
    
    tf_dirname = 'data/post_preprocessing/tf/'
    lv_dirname = 'data/post_preprocessing/lv/'
    
    out_dirname_base = 'data/kernels' + ('_v2' if kernel_v2_enabled else '')
    out_dirname_base += {
        kernel.OrderingStrategy.ARBITRARY: '_arb',
        kernel.OrderingStrategy.KEEP_FIRST: '_first',
        kernel.OrderingStrategy.KEEP_LAST: '_last',
        kernel.OrderingStrategy.PUSH_FRONT: '',  # default 
        kernel.OrderingStrategy.PUSH_BACK: '_back',
    }[ordering_strategy]

    tf_out_dirname = f'{out_dirname_base}/tf/'
    lv_out_dirname = f'{out_dirname_base}/lv/'
    

    # run tf graphs
    print('\nStarting kernelization on tf graphs')
    run(tf_dirname, tf_out_dirname, 
         first_seed, last_seed,
         target_kdistinct, k_orig_max, k_distinct_max, k_guess_max, w_iter_max, kernel_v2_enabled, ordering_strategy)
    
    # run lv graphs
    print('\nStarting kernelization on lv graphs')
    run(lv_dirname, lv_out_dirname, 
         first_seed, last_seed,
         target_kdistinct, k_orig_max, k_distinct_max, k_guess_max, w_iter_max, kernel_v2_enabled, ordering_strategy)
    
if __name__=="__main__":
    main()
    
    
    
    
