
import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Process # for timeout
import math, os, time, random, argparse, signal, pickle, re, copy

import algs.utils as utils
import algs.utils_misc as utils_misc
import algs.kernel as kernel

# different alg versions
import algs.bsd_dw as bsd_dw
import algs.bswd_dw_lp as bswd_dw_lp
import algs.bswd_dw_ip as bswd_dw_ip


def get_num_unrun_files(files, prerun_files, out_dirname, pp_dirname, k_distinct_max):
    '''
    Returns the number of files in the kernel dir that need running
    '''
    orig_unruncount=0
    distinct_unruncount=0
    guess_unruncount=0
    
    orig_count=0
    distinct_count=0
    guess_count=0
    
    for fname in files:
        fn = fname.split('/')[-1]
        ppfname = pp_dirname+fn  # post_preprocessing filename
        pklfname = fname
        
        # get post_preprocessing pkl file info
        with open(ppfname, 'rb') as infile: 
            pp_output = pickle.load(infile)
                    
        # get kernel pkl file info
        with open(pklfname, 'rb') as infile: 
            kernel_output = pickle.load(infile)
            
        k_distinct = pp_output['post_preprocessing']['kdistinct']
        if k_distinct>k_distinct_max:
            continue
                    
        if kernel_output['true_total_kernel'] is not None:
            if kernel_output['true_total_kernel']['passed_kernel']==True:
                orig_unruncount+=1
        
        if kernel_output['true_distinct_kernel'] is not None:
            if kernel_output['true_distinct_kernel']['passed_kernel']==True:
                distinct_unruncount+=1
        
        if kernel_output['guess_kernels'] is not None:
            if kernel_output['guess_kernels'][-0.6] is not None:
                guess_unruncount+=1
        
        # if graph has already been ran 
        partial_decompdat=None
        decomp_fn = out_dirname+fname.split('/')[-1] #[0:-16]+'_decomp.pkl'
        if decomp_fn in prerun_files:
            with open(decomp_fn, 'rb') as infile: 
                partial_decompdat = pickle.load(infile)
        
        if partial_decompdat != None:
            if partial_decompdat['decomp_data']['bsd_dw']['true_total']!=None:
                orig_count+=1
            
            if partial_decompdat['decomp_data']['bswd_dw_lp']['true_distinct']!=None:
                distinct_count+=1
                
            #NOTE since larger kvalue than true will always pass kernel
            if partial_decompdat['decomp_data']['bswd_dw_lp']['guesses'][-0.6]!=None:
                guess_count+=1
                
    counts = {'orig_unruncount' : orig_unruncount, 
              'distinct_unruncount' : distinct_unruncount,
              'guess_unruncount' : guess_unruncount,
              'orig_count' : orig_count,
              'distinct_count' : distinct_count,
              'guess_count' : guess_count}
    return counts


def get_final_clique_sets(B, vertex_indices, k_input):
    '''
    
    '''
    cliques = [[] for i in range(k_input)]
    
    i=0
    for row in B:
        j=0
        for c in row:
            if c==1:
                cliques[j].append(vertex_indices[i])
            j+=1
        i+=1
    return cliques


def compare_found_cliques(computed_cliqs, groundtruth_cliqs):
    '''
    Compares ground truth cliques to the computed cliques via decomp
    '''
    counted = []
    found_count=0
    for cliq in computed_cliqs:
        for gtcliq in groundtruth_cliqs:
            if sorted(cliq)==gtcliq and sorted(cliq) not in counted:
                counted.append(sorted(cliq))
                found_count+=1
    
    return found_count


def print_output_status(passed_bsd, time_bsd, reconstructs, found_cliq_fraction):
    CGREEN = '\33[42m'
    OKBLUE='\033[94m'
    CRED = '\033[91m'
    
    END = '\033[0m'
    
    if passed_bsd=='PASSED':
        col = CGREEN
    elif passed_bsd=='TIMEOUT':
        col = OKBLUE
    elif passed_bsd=='FAILED':
        col = CRED
    
    print(col+'END DECOMP: passed_bsd? '+passed_bsd+' time: '+str(time_bsd)+' reconstructs? '+str(reconstructs)+' found cliq frac: '+str(found_cliq_fraction)+END)
        

def run_bsd(A, k_input, alg_version, kernel_output, timeout, vertex_indices, groundtruth_cliqs, winf=None,
            perf_lp_v2_enabled=False, perf_lp_v3_enabled=False, num_threads=None):
    '''

    '''
    print('\nRunning decomp:', alg_version, ' kinput=', k_input)
    
    #---------------------------------------------- run bsd 
    tout = False
    B=np.full((1,1), -1)
    W=None
    num_lp_runs = 0
    
    # for the timeout
    signal.signal(signal.SIGALRM, utils_misc.handler) # WARNING only on unix?
    signal.alarm(timeout)
    start = time.time()
    try:
        if alg_version == 'bsd_dw':  
            # original BSD_DW
            B = bsd_dw.BSD_DW(A, k_input, winf) 
        elif alg_version == 'bswd_dw_lp':  
            # LP-weights
            B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A, k_input, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads)
        elif alg_version == 'bswd_dw_ip': 
            # int. part. weights (basis only)
            B, W = bswd_dw_ip.BSWD_DW(A, k_input)
    except utils_misc.TimeOutException as ex:
        print(ex)
        tout = True
    end = time.time()
    time_bsd = end-start
    signal.alarm(0)
    #----------------------------------------------

    reconstructs=None
    passed_bsd=None
    computed_cliques=None
    found_cliq_fraction=None
    
    if np.all(np.equal(B, -1)):    
        reconstructs=False
        if not tout:
            passed_bsd='FAILED'   # BSD outputs a no answer
        elif tout:
            passed_bsd='TIMEOUT'  # BSD_DW timed out
    else:
        passed_bsd='PASSED'       # BSD_DW found solution
            
        if alg_version=='bsd_dw':
            A_prime = np.dot(B, B.T)
        else:
            A_prime = np.dot(np.dot(B, W), B.T)
        A_masked = np.ma.masked_array(A, A==np.inf)  
        reconstructs = np.all(A_masked.astype(int)==A_prime)
        computed_cliques = get_final_clique_sets(B, vertex_indices, k_input)

        found_count = compare_found_cliques(computed_cliques, groundtruth_cliqs)
        found_cliq_fraction = found_count/len(groundtruth_cliqs)
        
    print_output_status(passed_bsd, time_bsd, reconstructs, found_cliq_fraction)
    
    out = {'B' : B,
           'W' : W,
           'kinput' : k_input,
           'found_cliq_fraction' : found_cliq_fraction,
           'alg_version' : alg_version,
           'time_bsd' : time_bsd,
           'passed_bsd' : passed_bsd,
           'reconstructs' : reconstructs,
           'vertex_indices' : vertex_indices,  
           'computed_cliques' : computed_cliques,
           'num_lp_runs': num_lp_runs}
    
    return out


def run_origncompare_exp(partial_decompdat, kernel_output):
    '''
    
    '''
    run_wecp=False
    run_ipart=False
    run_lp=False
    
    if kernel_output['true_total_kernel'] is not None:
        if kernel_output['true_total_kernel']['passed_kernel']:
        
            if partial_decompdat['bsd_dw']['true_total'] is None:
                run_wecp=True 
            
            if partial_decompdat['bswd_dw_lp']['true_total'] is None:
                run_lp=True 
            
            if partial_decompdat['bswd_dw_ip']['true_total'] is None:
                run_ipart=True 
        
    return run_wecp, run_ipart, run_lp
    

def run_distinct_exp(partial_decompdat, kernel_output, witer, k_distinct):
    '''
    
    '''
    run_ipart=False
    run_lp=False
    
    if kernel_output['true_distinct_kernel'] is not None:
        if kernel_output['true_distinct_kernel']['passed_kernel']==True:
            
            if partial_decompdat['bswd_dw_lp']['true_distinct'] is None:
                run_lp=True 
            
            if partial_decompdat['bswd_dw_ip']['true_distinct'] is None:
                run_ipart=True 
    
    if witer is not None and k_distinct is not None:
        if witer!=0 and k_distinct>=8:  # dont test all witers after 8
            run_lp=False
            run_ipart=False
                
    return run_ipart, run_lp
            
            
def run_guess_exp(partial_decompdat, kernel_output, k_distinct):
    '''
    
    '''
    run_iparts = {-0.6 : False, -0.4 : False, -0.2 : False, 
                   0.2 : False,  0.4 : False,  0.6 : False }
    
    run_lps   = {-0.6 : False, -0.4 : False, -0.2 : False, 
                  0.2 : False,  0.4 : False,  0.6 : False }
    
    guessout = kernel_output['guess_kernels']
    
    for key, value in partial_decompdat['bswd_dw_lp']['guesses'].items():
        if guessout is not None:
            if guessout[key] is not None:
                if guessout[key]['passed_kernel']:
                    if value is None:
                        run_lps[key]=True
    
    for key, value in partial_decompdat['bswd_dw_ip']['guesses'].items():
        if guessout is not None:
            if guessout[key] is not None:
                if guessout[key]['passed_kernel']:
                    if value is None:
                        run_iparts[key]=True
        
    if k_distinct >= 8:
            run_iparts[-0.6] = False
            run_iparts[-0.4] = False
            run_iparts[-0.2] = False
            run_iparts[0.6] = False
            run_iparts[0.4] = False
            run_iparts[0.2] = False
            
            run_lps[-0.6] = False
            run_lps[-0.4] = False
            run_lps[-0.2] = False
            run_lps[0.6] = False
            run_lps[0.4] = False
            run_lps[0.2] = False
            
    return run_iparts, run_lps
    


def run(pp_dirname, kern_dirname, out_dirname, first_seed, last_seed, k_distinct_max, timeout,
        target_kdistinct, lp_only, kernel_v2_enabled, ordering_strategy, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads):
    '''
    
    '''
    print('kernel in dirname:             ', kern_dirname)
    print('post_preprocessing in dirname: ', pp_dirname)
    print('out_dirname:                   ', out_dirname)
    
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname, exist_ok=True)
        
    kernel_files = utils_misc.get_files(kern_dirname, '.pkl')
    files = utils_misc.filter_files_seed(kernel_files, first_seed, last_seed)

    # already processed graphs
    prerun_files = utils_misc.get_files(out_dirname, '.pkl') 
    
    counts = get_num_unrun_files(files, prerun_files, out_dirname, pp_dirname, k_distinct_max)
    count=0
    
    for fname in files: 
        print('\n____________________________________________ {}/{} : {}'.format(count, 
                                                                       len(files),
                                                                       fname))
        print('{}/{} orig exp, {}/{} distinct exp, {}/{} guess exp'\
            .format(counts['orig_count'], counts['orig_unruncount'], 
                    counts['distinct_count'], counts['distinct_unruncount'], 
                    counts['guess_count'], counts['guess_unruncount']))
        count+=1
        
        # if graph has already been ran 
        partial_decompdat=None
        decomp_fn = out_dirname+fname.split('/')[-1] #+'.pkl'
        if decomp_fn in prerun_files:
            with open(decomp_fn, 'rb') as infile: 
                partial_decompdat = pickle.load(infile)
                
        # get post_preprocessing pkl file info
        ppfname = pp_dirname+fname.split('/')[-1]  # post_preprocessing filename
        with open(ppfname, 'rb') as infile: 
            pp_output = pickle.load(infile)
        
        # get kernel pkl file info
        pklfname = fname
        with open(pklfname, 'rb') as infile: 
            kernel_output = pickle.load(infile)
                
        # collect + write out data  
        decomp_data=None
        if partial_decompdat is None:
            decomp_data = {
                'timeout' : timeout,
                'bsd_dw'     : {'true_total'    : None},
                'bswd_dw_lp' : {'true_total'    : None,
                                'true_distinct' : None,
                                'guesses'       :  {-0.6 : None, 
                                                    -0.4 : None,
                                                    -0.2 : None, 
                                                    0.2 : None, 
                                                    0.4 : None,
                                                    0.6 : None }},
                'bswd_dw_ip' : {'true_total'    : None,
                                'true_distinct' : None,
                                'guesses'       :  {-0.6 : None, 
                                                    -0.4 : None,
                                                    -0.2 : None, 
                                                    0.2 : None, 
                                                    0.4 : None,
                                                    0.6 : None }}}
        else:
            decomp_data = partial_decompdat['decomp_data']
        
        k_total = pp_output['post_preprocessing']['ktotal']
        k_distinct = pp_output['post_preprocessing']['kdistinct']
        
        print('ktotal ', k_total, ' kdistinct', k_distinct)
        if target_kdistinct is None and k_distinct>k_distinct_max:
            print('k_distinct > k_distinct_max....skipping for now')
            continue
        if target_kdistinct is not None and k_distinct != target_kdistinct:
            print('skipping because kdistinct != target_kdistinct')
            continue
        
        witer=None 
        if 'witer' in fname:
            witer = int(utils_misc.get_fname_value(fname, 'witer'))
                                
        #------------------------------------------------- run true total 
        run_wecp, run_ipart, run_lp = run_origncompare_exp(decomp_data, kernel_output)
            
        if run_wecp and not lp_only: #or run_ipart or run_lp:
            '''
            need to run orig, ip, and lp versions w. true total 
            '''
            print('\n---------------------run true total')
            if kernel_output['true_total_kernel'] is None:
                print('ERROR: must run kernel first')
            
            counts['orig_count']+=1
            A = kernel_output['true_total_kernel']['A_kernel']
            vertex_indices = list(A.index.values)
            groundt_cliqs = kernel_output['true_total_kernel']['clique_vertices']
            
            k_total = kernel_output['true_total_kernel']['kinput']
            winf = kernel_output['true_total_kernel']['w_inf']
            A = A.to_numpy()
            
        if run_wecp and not lp_only:
            decomp_data['bsd_dw']['true_total'] = run_bsd(A, k_total, 'bsd_dw',
                                                          kernel_output, 
                                                          timeout, vertex_indices, 
                                                          groundt_cliqs, winf=winf)
            
                  
        #------------------------------------------------- run true distinct
        run_ipart, run_lp = run_distinct_exp(decomp_data, kernel_output, witer, k_distinct)
        
        if run_lp or run_ipart:
            '''
            run ip and lp versions w. true_distinct
            '''
            print('\n---------------------run true distinct')
            counts['distinct_count']+=1
            A = kernel_output['true_distinct_kernel']['A_kernel']
            vertex_indices = list(A.index.values)
            groundt_cliqs = kernel_output['true_distinct_kernel']['clique_vertices']
            k_distinct = kernel_output['true_distinct_kernel']['kinput']
            A = A.to_numpy()
        
        if run_lp:
            decomp_data['bswd_dw_lp']['true_distinct'] = run_bsd(A, k_distinct, 'bswd_dw_lp',
                                                                 kernel_output, timeout, 
                                                                 vertex_indices, groundt_cliqs,
                                                                 perf_lp_v2_enabled=perf_lp_v2_enabled,
                                                                 perf_lp_v3_enabled=perf_lp_v3_enabled,
                                                                 num_threads=num_threads)
                    
        if run_ipart and not lp_only:
            decomp_data['bswd_dw_ip']['true_distinct'] = run_bsd(A, k_distinct, 'bswd_dw_ip',
                                                                 kernel_output, timeout, 
                                                                 vertex_indices, groundt_cliqs)
                                
                                
        #------------------------------------------------- run guesses 
        run_iparts, run_lps = run_guess_exp(decomp_data, kernel_output, k_distinct)
        
        if True in run_iparts.values() or True in run_lps.values():
            '''
            run ip and lp versions w. all guess k values
            '''
            print('\n---------------------run guesses')
            counts['guess_count']+=1
        
        for guess, torun in run_iparts.items():
            # if the kdistinct value is specified, enable guesses unless lp-only is not specified
            if (target_kdistinct is not None or torun) and not lp_only: 
                A = kernel_output['guess_kernels'][guess]['A_kernel']
                vertex_indices = list(A.index.values)
                groundt_cliqs = kernel_output['guess_kernels'][guess]['clique_vertices']
                k_guess = kernel_output['guess_kernels'][guess]['kinput']
                A = A.to_numpy() 

                decomp_data['bswd_dw_ip']['guesses'][guess] = run_bsd(A, k_guess, 'bswd_dw_ip', 
                                                                      kernel_output, timeout, 
                                                                      vertex_indices, groundt_cliqs)
        
        for guess, torun in run_lps.items():
            # if the kdistinct value is specified, always enable guesses
            if target_kdistinct is not None or torun:
                # if you get an exception here, make sure that you have run exp_runkernel.py
                # with appropriate options (--kdistinct, --v2, --ordering).
                A = kernel_output['guess_kernels'][guess]['A_kernel']
                vertex_indices = list(A.index.values)
                groundt_cliqs = kernel_output['guess_kernels'][guess]['clique_vertices']
                k_guess = kernel_output['guess_kernels'][guess]['kinput']
                A = A.to_numpy() 
                            
                decomp_data['bswd_dw_lp']['guesses'][guess] = run_bsd(A, k_guess, 
                        'bswd_dw_lp', kernel_output, timeout, 
                        vertex_indices, groundt_cliqs,
                        perf_lp_v2_enabled=perf_lp_v2_enabled,
                        perf_lp_v3_enabled=perf_lp_v3_enabled,
                        num_threads=num_threads)
            
        #------------------------------------------------- final output data
        if partial_decompdat is None:
            decomp_output = { 
                'decomp_data' : decomp_data
                }
        else:
            decomp_output = partial_decompdat

        # options for v2.0.0
        decomp_output['kernel_v2'] = kernel_v2_enabled
        decomp_output['ordering'] = ordering_strategy
        decomp_output['perf_lp_v2'] = perf_lp_v2_enabled
        decomp_output['perf_lp_v3'] = perf_lp_v3_enabled
        decomp_output['num_threads'] = num_threads

        with open(decomp_fn, 'wb') as f:    
            pickle.dump(decomp_output, f)


def main():
    parser = argparse.ArgumentParser()
   
    #----------------------------------- args
    parser.add_argument('-f', '--first_seed', type=int,
        help="enter first seed value in seed range", required=True)
    parser.add_argument('-l', '--last_seed', type=int,
        help="enter last seed value in seed range", required=True) 
 
    parser.add_argument('-o', '--timeout', type=int, default=3600, # 1 hr default
        help="stop BSD after timeout number of seconds", required=False)    

    # since v2.0.0
    parser.add_argument('--kdistinct', type=int,
        help='limit experiments to the instances with the specific kdistinct value', required=False)
    parser.add_argument('--lp-only', action='store_true',
        help='run algorithm lp only (default: False)')
    parser.add_argument('--kernel-v2', action='store_true',
        help='use kernels created with the v2 option (default: False)')
    parser.add_argument('--ordering', choices=['arbitrary', 'keep_first', 'keep_last', 'push_front', 'push_back'],
        help='vertex reordering strategy (default: push_front)', default = 'push_front')
    parser.add_argument('--perf-lp-v2', action='store_true',
        help='use performance tuning techniques for algorithm lp (default: False)')
    parser.add_argument('--perf-lp-v3', action='store_true',
        help='use performance tuning techniques v3 for algorithm lp (default: False)')
    parser.add_argument('-t', '--num-threads', type=int,
        help="number of threads used for Gurobi (default: Gurobi's default value)")

    args = vars(parser.parse_args())
    
    first_seed = args.get('first_seed')
    last_seed = args.get('last_seed')
    timeout = args.get('timeout')    
    target_kdistinct = args.get('kdistinct')
    lp_only = args.get('lp_only')
    kernel_v2_enabled = args.get('kernel_v2')
    ordering_strategy = {
        'arbitrary': kernel.OrderingStrategy.ARBITRARY,
        'keep_first': kernel.OrderingStrategy.KEEP_FIRST,
        'keep_last': kernel.OrderingStrategy.KEEP_LAST,
        'push_front': kernel.OrderingStrategy.PUSH_FRONT,
        'push_back': kernel.OrderingStrategy.PUSH_BACK,
    }[args['ordering']]
    perf_lp_v2_enabled = args.get('perf_lp_v2')
    perf_lp_v3_enabled = args.get('perf_lp_v3')
    num_threads = args.get('num_threads')
    
    k_orig_max=11
    k_distinct_max=11
    k_guess_max=7
    w_iter_max=5
    
    kernel_dir_suffix = '_v2' if kernel_v2_enabled else ''
    ordering_suffix = {
        kernel.OrderingStrategy.ARBITRARY: '_arb',
        kernel.OrderingStrategy.KEEP_FIRST: '_first',
        kernel.OrderingStrategy.KEEP_LAST: '_last',
        kernel.OrderingStrategy.PUSH_FRONT: '',  # default 
        kernel.OrderingStrategy.PUSH_BACK: '_back',
    }[ordering_strategy]
    tf_kern_dirname = f'data/kernels{kernel_dir_suffix}{ordering_suffix}/tf/'
    lv_kern_dirname = f'data/kernels{kernel_dir_suffix}{ordering_suffix}/lv/'
    
    tf_pp_dirname = 'data/post_preprocessing/tf/'
    lv_pp_dirname = 'data/post_preprocessing/lv/'
    
    out_dir_suffix = {
        (True, True, False): f'_kv2{ordering_suffix}_pv2',
        (True, False, False): f'_kv2{ordering_suffix}_pv1',
        (False, True, False): f'_kv1{ordering_suffix}_pv2',
        (False, False, False): f'{ordering_suffix}',
        (True, True, True): f'_kv2{ordering_suffix}_pv3',
        (True, False, True): f'_kv2{ordering_suffix}_pv3',
        (False, True, True): f'_kv1{ordering_suffix}_pv3',
        (False, False, True): f'_kv1{ordering_suffix}_pv3',
    }[(kernel_v2_enabled, perf_lp_v2_enabled, perf_lp_v3_enabled)]

    tf_out_dirname = f'data/finaldata{out_dir_suffix}/tf/'
    lv_out_dirname = f'data/finaldata{out_dir_suffix}/lv/'
    
    # run bsd on tf graphs
    print('\nStarting bsd algorithms on tf kernels')
    run(tf_pp_dirname, tf_kern_dirname, tf_out_dirname, 
        first_seed, last_seed, k_distinct_max, timeout,
        target_kdistinct, lp_only, kernel_v2_enabled, ordering_strategy, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads)
    
    # run bsd on lv graphs
    print('\n\nStarting bsd algorithms on lv kernels')  
    run(lv_pp_dirname, lv_kern_dirname, lv_out_dirname, 
        first_seed, last_seed, k_distinct_max, timeout,
        target_kdistinct, lp_only, kernel_v2_enabled, ordering_strategy, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads)


if __name__=="__main__":
    main()
    
    
