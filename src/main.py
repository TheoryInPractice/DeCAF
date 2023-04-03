
'''
    Script to run wecp, ipart, or lp on single graph files. 
    Accepted graph file formats: .csv and .txt edge lists
'''

__version__ = '2.0.0'
__lisence__ = 'BSD-3-Clause'

import argparse, time
import networkx as nx
import numpy as np

import algs.utils as utils
import algs.utils_misc as utils_misc

import algs.preprocess as preprocess
import algs.kernel as kernel

# different alg versions
import algs.bsd_dw as bsd_dw
import algs.bswd_dw_lp as bswd_dw_lp
import algs.bswd_dw_ip as bswd_dw_ip


def read_data(file_name):
    '''
    if file format is .csv, reads in each line
    if file format is .txt, reads graph directly
    '''

    G = nx.Graph()
    if file_name[-4:] == '.csv':
        with open(file_name) as fn:
            lines = fn.readlines()
            
            for line in lines:
                ln = line.split(',')
                
                if len(ln) == 3:
                    try:
                        u = int(ln[0])
                        v = int(ln[1])
                        w = int(ln[2])
                        
                        G.add_edge(u, v, weight=w)
                        
                    except:
                        pass
                else:
                    print("Error reading data file")
                
    elif file_name[-4:] == '.txt':
        G = nx.read_weighted_edgelist(file_name, nodetype=int)
    else:
        print('Error: incorrect file format')
        
    return G
        

def run_preprocess(G, k_distinct):
    '''
    runs preprocessor to remove disjoint cliques/single overlapping cliques
    '''
    n_pre = G.order()
    start = time.time()
    dat = preprocess.preprocess(G, k_distinct)
    end = time.time()   
    time_proc = end - start        
    
    n_post = G.order()
    k_post_pp = dat['kdistinct_postproc']
    
    print('preprocess time: {}, n_pre={}, n_post={}'.format(time_proc, n_pre, n_post))
    
    return G, k_post_pp
    

def run_kernel(A, G, k, kernel_v2_enabled, ordering_strategy):
    '''
    kernelizes graph

    Returns: Reduced adjacency matrix
             None if the instance is infeasible
    '''
    n_pre = G.order()
    
    # reduction rules    
    start = time.time()
    A_kernel, removal_vertices, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A, k, kernel_v2_enabled, ordering_strategy)
    end = time.time()   
    time_kernel = end - start
    
    # check if the instance is found to be infeasible
    if removal_vertices != [-1]:
        G.remove_nodes_from(removal_vertices)
        n_post = G.order()
        
        print('kernel time: {}, n_pre={}, n_post={}'.format(time_kernel, n_pre, n_post))
        return A_kernel
    else:
        print('kernel time: {}, infeasible'.format(time_kernel))
        return None


def run_bsd(A, G, k_input, alg_version, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads):
    '''
    runs either wecp, ipart, or lp decomposition algorithms
    '''
    W=None
    num_lp_runs = 0
    start = time.time()
    if alg_version == 'wecp':  
        # original BSD_DW (Feldman et al. 2020)
        if G.order()>0:
            winf =  np.linalg.norm(nx.to_numpy_matrix(G), np.inf)
        
        B = bsd_dw.BSD_DW(A, k_input, winf) 
    elif alg_version == 'lp':  
        # linear programming version
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A, k_input, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads)
    elif alg_version == 'ipart': 
        # integer partitioning version
        B, W = bswd_dw_ip.BSWD_DW(A, k_input)
    else:
        print('Error: incorrect algorithm version. Options: "wecp", "lp", "ipart"')
            
    end = time.time()
    time_bsd = end-start
    print(f'Decomposition time: {time_bsd}, num_lp_runs={num_lp_runs}')
    
    return B, W


def run(alg_version, G, k, to_preprocess, to_kernelize, kernel_v2_enabled, ordering_strategy, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads):
    
    if to_preprocess:
        G, k = run_preprocess(G, k)
    
    A = utils_misc.get_wildcard_adjacency(G)
    if to_kernelize:
        A = run_kernel(A, G, k, kernel_v2_enabled, ordering_strategy)
        if A is None:
            print("Decomposition Failed")
            return

    A = A.to_numpy()
    B, W = run_bsd(A, G, k, alg_version, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads)

    print('\nClique Membership Matrix B:\n', B)  # FIXME: This may not output the whole results
    print('Weight Matrix W:\n', W)
    
    if np.all(np.equal(B, -1)):  
        print("Decomposition Failed")
    else:
        print("Decomposition Passed")
        
        if alg_version=='wecp':
            A_prime = np.dot(B, B.T)
        else:
            A_prime = np.dot(np.dot(B, W), B.T)
        A_masked = np.ma.masked_array(A, A==np.inf)  
        reconstructs = np.all(A_masked.astype(int)==A_prime)
        print("Reconstructs solution? ", reconstructs)


def get_parser():
    '''
    Argument parser.
    '''
    parser = argparse.ArgumentParser(description='Script to run wecp, ipart, or lp on single graph files.')

    #----------------------------------- required args
    parser.add_argument('-a', '--algorithm', type=str,
        help="options: wecp, ipart, lp", required=True)
    
    parser.add_argument('-g', '--graph_filename', type=str,
        help="enter .txt edge list data file", required=True)
    
    parser.add_argument('-k', '--parameter', type=int,
        help="enter integer parameter value", required=True)
    
    #----------------------------------- nonrequired args
    parser.add_argument('-p', '--preprocess', action='store_true',
        help="enable preprocessing (default: False)", required=False)
    parser.add_argument('--skip-kernelization', action='store_true',
        help="skip kernelization (default: False)", required=False)

    #----------------------------------- since v2.0.0
    parser.add_argument('--profile', action='store_true',
        help='enable profiler (default: False)')
    parser.add_argument('--kernel-v2', action='store_true',
        help='use an improved kernelization technique based on the nonuniform Fisher inequality (default: False)')
    parser.add_argument('--ordering', choices=['arbitrary', 'keep_first', 'keep_last', 'push_front', 'push_back'],
        help='vertex reordering strategy (default: push_front)', default = 'push_front')
    parser.add_argument('--perf-lp-v2', action='store_true',
        help='use performance tuning techniques for algorithm lp (default: False)')
    parser.add_argument('--perf-lp-v3', action='store_true',
        help='use performance tuning techniques v3 for algorithm lp (default: False)')
    parser.add_argument('-t', '--num-threads', type=int,
        help="number of threads used for Gurobi (default: Gurobi's default value)")
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')

    return parser


def main(args):
    ordering_strategy = {
        'arbitrary': kernel.OrderingStrategy.ARBITRARY,
        'keep_first': kernel.OrderingStrategy.KEEP_FIRST,
        'keep_last': kernel.OrderingStrategy.KEEP_LAST,
        'push_front': kernel.OrderingStrategy.PUSH_FRONT,
        'push_back': kernel.OrderingStrategy.PUSH_BACK,
    }[args.ordering]

    print("Filename : ", args.graph_filename)
    G = read_data(args.graph_filename)
    run(args.algorithm, G, args.parameter, args.preprocess, not args.skip_kernelization,
        args.kernel_v2, ordering_strategy, args.perf_lp_v2, args.perf_lp_v3, args.num_threads)


if __name__=="__main__":
    args = get_parser().parse_args()
    if args.profile:
        # enable profiler
        import cProfile
        cProfile.run('main(args)')
    else:
        # run without profiler
        main(args)
