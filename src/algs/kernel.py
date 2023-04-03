'''
    Kernel reduction rules 
'''

import numpy as np
import algs.utils as utils
from enum import Enum


class OrderingStrategy(Enum):
    """
    Strategy to reorder vertices after kernelization.
    """
    ARBITRARY = 1  # keep an arbitrary vertex in a reduced block
    KEEP_FIRST = 3  # keep the first (in the adjaceny matrix) vertex in a reduced block
    KEEP_LAST = 4  # keep the last (in the adjaceny matrix) vertex in a reduced block
    PUSH_FRONT = 5  # move an arbitrary vertex in a reduced block to the front of the adjacency matrix
    PUSH_BACK = 6  # move an arbitrary vertex in a reduced block to the back of the adjacency matrix

def num_blocks(A):
    '''
    Args: A : 2-D pandas adjacency list
    '''    
    # reduction rule 1
    blocks=[]
    inds = list(A.index.values)
    added = {}
    for ind in inds:
        added[ind]=False
    count=0
        
    for i in inds:
        block=set()
        if added[i]==False:  # current vert i has not been added to a block yet
            block.add(i)
            added[i]=True
            
            # try to find all of i's equiv rows
            for j in inds[count:]:
                if i != j and added[j]==False: # if j has not been added yet to a block    
                    if utils.equiv_rows(A.loc[i, :], A.loc[j, :]) and A.loc[i, j]>=1:
                        block.add(j)
                        added[j]=True
            blocks.append(block)
        count+=1
         
    return blocks


def remove_arbitrary(A, blocks, k, kernel_v2_enabled):
    '''
    Args: A : 2-D pandas adjacency list
    '''    
    # reduction rule 2
    toremove = []
    b_index=0
    num_reduced_blocks = 0  # the number of blocks the reduction rule was applicable to

    # k: smaller kernel based on the nonuniform Fisher inequality
    block_size_threshold = k if kernel_v2_enabled else 2 ** k
        
    for block in blocks:
        if len(block) > block_size_threshold:
            v = block.pop()
            u = block.pop()
            A.loc[v,v] = A.loc[u,v]
            
            toremove.append(u)            

            for i in block:
                toremove.append(i)
  
            # replace block w. block containing representative vertex
            newblock = set()
            newblock.add(v)
            blocks[b_index]=newblock

            num_reduced_blocks += 1
        
        b_index+=1
    
    A_red = A.drop(toremove, axis=1)
    A_red = A_red.drop(toremove, axis=0)
        
    return A_red, toremove, num_reduced_blocks


def remove_and_reorder(A, blocks, k, kernel_v2_enabled, ordering_strategy):
    '''
    Removes vertices in large blocks except one representative vertex,
    set vertex weights, and moves the representative vertices to the
    front of the adjacency matrix to for better performance.

    Args: A : 2-D pandas adjacency list
    '''    
    # reduction rule 2
    toremove = []
    representatives = []
    num_reduced_blocks = 0  # the number of blocks the reduction rule was applicable to

    # k: smaller kernel based on the nonuniform Fisher inequality
    block_size_threshold = k if kernel_v2_enabled else 2 ** k

    for block in blocks:
        if len(block) > block_size_threshold:
            if ordering_strategy == OrderingStrategy.KEEP_FIRST:
                v = min((A.index.get_loc(x), x) for x in block)[1]
                block.remove(v)
            elif ordering_strategy == OrderingStrategy.KEEP_LAST:
                v = max((A.index.get_loc(x), x) for x in block)[1]
                block.remove(v)
            else:
                # arbitrarily choose the representative vertex in the block
                v = block.pop()

            u = block.pop()
            A.loc[v,v] = A.loc[u,v]  # set weight to the representative vertex
            
            toremove += [u] + list(block)
            representatives += [v]
            num_reduced_blocks += 1

    # remove vertices
    A_red = A.drop(toremove, axis=1)
    A_red = A_red.drop(toremove, axis=0)

    # reorder indices and columns
    if representatives and ordering_strategy in [OrderingStrategy.PUSH_FRONT, OrderingStrategy.PUSH_BACK]:
        if ordering_strategy == OrderingStrategy.PUSH_FRONT:
            # representatives come first
            # FIXME: do this in O(n)
            permutation = representatives + [i for i in A_red.index if i not in representatives]
        elif ordering_strategy == OrderingStrategy.PUSH_BACK:
            # representatives come last
            permutation = [i for i in A_red.index if i not in representatives] + representatives
        else:
            assert False, 'never happens'
        A_red = A_red.reindex(index=permutation, columns=permutation)

    return A_red, toremove, num_reduced_blocks


def reduction_rules(A, k, kernel_v2_enabled, ordering_strategy=OrderingStrategy.PUSH_FRONT):
    '''
    
    '''
    blocks = num_blocks(A)
    num_total_blocks = len(blocks)

    if num_total_blocks > 2**k:
        return A, [-1], num_total_blocks, 0  # NO instance
    
    if ordering_strategy == OrderingStrategy.ARBITRARY:
        A_red, removed_vertices, num_reduced_blocks = remove_arbitrary(A, blocks, k, kernel_v2_enabled)
    else:
        A_red, removed_vertices, num_reduced_blocks = remove_and_reorder(A, blocks, k, kernel_v2_enabled, ordering_strategy)
    
    return A_red, removed_vertices, num_total_blocks, num_reduced_blocks


