'''
    Binary Symmetric Weighted Decomposition w. Diagonal Wildcards
    Finds the weights by solving a LP (only on basis rows)
    Computes basis by backtracking
    
    (lp algorithm)
'''   

from collections import defaultdict
import random, math 
from typing import Dict, Union, List, Tuple, Optional
import numpy as np
import copy

import gurobipy as gp
from gurobipy import GRB

import algs.utils as utils


class Bookeeper:
    def __init__(self, k, max_val):
        self.max_val = max_val
        self.k = k
        self.b = 0
        self.i = 0
        
        # for extending basis -- gives each pattern row 
        self.cart_V = utils.PatternRows(k)
        
        # each index represents a basis row, the value represents the 
        # index in cartV of the next pattern, each time we need a new 
        # pattern row, current is returned then index is incremented 
        self.next_basis_pattern = [0]*max_val  
        
        # tracks the indices in B that are filled w. basis rows
        self.basis_inds = [-1]*max_val  
            
        # holds num constraints added to insert the basis row at each index
        self.cnstrs_added = [0]*max_val    
        
        # at index i, holds the weight matrix found before inserting basis row i
        self.Ws = [0]*max_val
                
    def is_completed(self) -> bool:
        return all(val == (2**self.k-1) for val in self.next_basis_pattern)

    def get_next_patt_ind(self):
        # returns the index of the desired pattern row, increments
        while True:
            if self.next_basis_pattern[self.b]>=2**self.k:
                return -1
            
            ind = self.next_basis_pattern[self.b]
            self.next_basis_pattern[self.b]+=1
        
            # ind = 0 -> [0, ..., 0] so we don't want this
            if ind != 0:  
                return ind

    def get_next_patt_row(self):
        ind = self.get_next_patt_ind()
    
        if ind == -1:
            return [-1]
        else:
            Pb = self.cart_V.entryAt(ind) # get potential basis row 
            return Pb
    
    
    def backtrack(self, B, model):
        '''
        called when self.b == 2**self.k, meaning
        there are no more unique pattern rows to try
        given the previous pattern rows, so backtrack 
        '''    
        if self.b-1==-1:
            return False
        
        self.next_basis_pattern[self.b]=0
        self.Ws[self.b]=0  # reset weights before inserting b
        
        self.b-=1  # go back to rethink last pattern row
        
        self.i = self.basis_inds[self.b]
        
        # remove constraints that allowed inserting of new self.b
        utils.remove_constraints(model, self.cnstrs_added[self.b])
        self.cnstrs_added[self.b]=0
        self.basis_inds[self.b] =- 1
        B[self.i] = [-1]*self.k   
        
        return True


class SignatureManager:
    """
    Manages a set of signatures grouped by blocks.
    """
    def __init__(self, block_ids: List[int]) -> None:
        self._block_ids = block_ids
        self._signature_to_block_id: Dict[int, int] = {}
        self._block_id_to_signatures: Dict[int, List[int]] = defaultdict(list)

    def copy(self):
        ret = SignatureManager(self._block_ids)
        ret._signature_to_block_id = copy.deepcopy(self._signature_to_block_id)
        ret._block_id_to_signatures = copy.deepcopy(self._block_id_to_signatures)
        return ret

    def get_identical_block_signature(self, v: int) -> Optional[int]:
        """
        Checks if the given vertex must be in an identical block.

        @return signature (integer representation of a bitset) if v must be in an identical block
                None otherwise
        """
        others = self._block_id_to_signatures[self._block_ids[v]]
        if len(others) >= 2 and others[0] == others[1]:
            # found identical twins
            return others[0]
        else:
            return None

    def is_valid(self, v: int, signature: int) -> bool:
        this_block_id = self._block_ids[v]

        if this_block_id < 0:
            # singleton; valid if the signature is unused
            return self._is_signature_unused(signature)
        else:
            # block member; check other members
            other_block_signatures = self._block_id_to_signatures[this_block_id]
            if len(other_block_signatures) == 0:
                # first member in this block; valid if the signature is unused
                return self._is_signature_unused(signature)
            elif len(other_block_signatures) == 1:
                # this vertex can be an identical twin
                if signature == other_block_signatures[0]:
                    return True
                # otherwise, it must be a fraternal twin
                return self._is_signature_unused(signature) and self._is_valid_fraternal(signature, other_block_signatures[0])
            else:
                if other_block_signatures[0] == other_block_signatures[1]:
                    # found identical twins
                    return signature == other_block_signatures[0]
                # otherwise, fraternal twins
                return self._is_signature_unused(signature) and all(self._is_valid_fraternal(signature, t) for t in other_block_signatures)

    def add_signature(self, v: int, signature: int) -> None:
        # assert self.is_valid(v, signature)  # use this for debugging

        block_id = self._block_ids[v]
        if block_id < 0:
            # singleton
            self._signature_to_block_id[signature] = block_id
        else:
            self._signature_to_block_id[signature] = block_id
            self._block_id_to_signatures[block_id] += [signature]

    def remove_signature(self, v: int, signature: int) -> None:
        block_id = self._block_ids[v]
        if block_id < 0:
            # singleton
            del self._signature_to_block_id[signature]
        else:
            # remove the first occurrence of this signature in this block
            self._block_id_to_signatures[block_id].remove(signature)

            if self._block_id_to_signatures[block_id]:
                if self._block_id_to_signatures[block_id][0] != signature:
                    # fraternal block
                    del self._signature_to_block_id[signature]
            else:
                # if this was the last one, remove keys
                del self._block_id_to_signatures[block_id]
                del self._signature_to_block_id[signature]

    def _is_signature_unused(self, signature: int) -> bool:
        return self._signature_to_block_id.get(signature) is None

    def _is_valid_fraternal(self, s: int, t: int) -> bool:
        """Returns true if `s` is not a subset of `t` and `t` is not a subset of  `s`."""
        return s & t != min(s, t)


class BookeeperV2:
    def __init__(self, k, max_val, block_ids):
        self.max_val = max_val
        self.k = k
        self.b = 0
        self.i = 0
        self.sig_mgr = SignatureManager(block_ids)

        # array of PatternIterators
        self.patterns = [None] * max_val

        # set True if the pattern for an identical block has already been examined
        self.identical_tried = [False] * max_val

        # tracks the indices in B that are filled w. basis rows
        self.basis_inds = [-1] * max_val

        # holds num constraints added to insert the basis row at each index
        self.cnstrs_added = [0] * max_val

        # at index i, holds the weight matrix found before inserting basis row i
        self.Ws = [0] * max_val

    def is_completed(self) -> bool:
        return all(p is not None and not p.has_next() for p in self.patterns)

    def get_next_patt_row(self, A) -> Tuple[Union[np.ndarray, List[int]], int]:
        # check if self.i is in an identical block
        ident_sig = self.sig_mgr.get_identical_block_signature(self.i)
        if ident_sig is not None:
            # row self.i must be identical to other block members
            if self.identical_tried[self.b]:
                # already tried
                return [-1], -1
            else:
                self.identical_tried[self.b] = True
                return utils.bitset_to_vector(ident_sig, self.k), ident_sig

        if self.patterns[self.b] is None:
            # create a new PatternIterator with the constraints for the current i
            forbidden = np.zeros(self.k, dtype=bool)

            for j in range(self.b):
                if self.identical_tried[j]:
                    # safe to ignore this case
                    # 
                    # If this check is true, then j's block is an identical block and must have at least
                    # two vertices self.basis_inds[u1] and self.basis_inds[u2] such that
                    # u1 < u2 < j and self.identical_tried[u1]==False.
                    # If self.i and self.basis_inds[j] are adjacent, the vector `forbidden` stays the same.
                    # Otherwise, self.i is not adjacent to u1 because u1 and j are in the same block.
                    # By definition, j's signature is equivalent to u1's signature and
                    # u1's signature is added to the forbidden vector before j is processed.
                    continue

                u = self.basis_inds[j]  # index in the basis matrix
                if A[self.i][u] == 0:
                    # no edge between the current i and (already-filled) u
                    forbidden = np.logical_or(forbidden, self.patterns[j].get())

            self.patterns[self.b] = utils.PatternIterator(forbidden)

        # get potential basis row
        while True:
            ret = self.patterns[self.b].next()
            if ret is None:
                # no potential basis row
                break

            sig = self.patterns[self.b].get_int()
            if self.sig_mgr.is_valid(self.i, sig):
                # valid basis row
                self.sig_mgr.add_signature(self.i, sig)
                return ret, sig
            
        return [-1], -1

    def backtrack(self, B, model) -> bool:
        '''
        called when self.b == 2**self.k, meaning
        there are no more unique pattern rows to try
        given the previous pattern rows, so backtrack 
        '''
        if self.b - 1 == -1:
            return False

        self.patterns[self.b] = None
        self.identical_tried[self.b] = False
        self.Ws[self.b] = 0  # reset weights before inserting b

        self.b -= 1  # go back to rethink last pattern row

        self.i = self.basis_inds[self.b]

        # roll back signature info
        if not self.identical_tried[self.b]:
            sig = self.patterns[self.b].get_int()
            self.sig_mgr.remove_signature(self.i, sig)

        # remove constraints that allowed inserting of new self.b
        utils.remove_constraints(model, self.cnstrs_added[self.b])
        self.cnstrs_added[self.b] = 0
        self.basis_inds[self.b] = - 1
        B[self.i] = [-1] * self.k

        return True


def optimize_model(model, book, W): 
    # Returns True is model is feasible+bounded and updates W
    # False otherwise
    
    # Solve model
    model.optimize()
    status = model.status
    
    # Get the variables
    # inserts weights in W diagonals
    if status == GRB.Status.OPTIMAL: 
        ind = 0
        for v in model.getVars():
            W[ind, ind]=v.x
            ind+=1
    elif status == GRB.Status.INF_OR_UNBD or status==GRB.Status.INFEASIBLE:
        return False
    
    return True


def add_model_constraints(A, B, Pi, book, dv_w, model):
    '''
    book.i: current index of B we are attempting to insert a basis row (Pi).
    book.basis_inds: the row indices of the previously added basis rows
    
    returns: number of constraints added at this one step.
    '''
    #---------- Add the new constraints to model and solve
    # diagonal constraints: 
    # Aii = Pi1*W1*Pi1 + Pi2*W2*Pi2 + ... + Pik*Wk*Pik
    count=0
    if A[book.i, book.i] != math.inf:
        aii = 0
        for q in range(book.k):
            aii += Pi[q] * dv_w[q] * Pi[q]
        
        cst_name = "{}constraint_{}".format(book.i,count)
        model.addConstr(aii == A[book.i,book.i], name=cst_name)
        count+=1
            
    # non-diagonal constraints: 
    # Aij = Pi1*W1*Bj1.T + Pi2*W2*Bj2.T + ... + Pik*Wk*Bjk.T
    for j in book.basis_inds:
        if j != -1 and A[book.i,j] != math.inf:  
            aij = 0
            for q in range(book.k):
                aij += Pi[q] * dv_w[q] * B[j, q].T
                        
            cst_name = "{}constraint_{}".format(book.i,count)
            model.addConstr(aij == A[book.i,j], name=cst_name)
            count+=1

    model.update()
    return count


def extend_basis(A, B, W, cart_V, k):
    row_i = 0
    
    for Bi in B:
        filled = False
        
        if np.all(Bi == -1):
            for j in range(0,2**k):
                v = np.array(cart_V.entryAt(j))
                if(utils.weighted_i_compatible(A, B, W, v, row_i)):
                    B[row_i] = v
                    filled=True
                    break
                
            if not filled:
                return B, row_i
        row_i+=1
    
    return B, A.shape[0] 


def extend_basis_v2(A, B, W, k, sig_mgr):
    row_i = 0

    # Note: do not use enumerate() for better performance
    for Bi in B:
        filled = False
        
        if np.all(Bi == -1):
            # check if row_i is in an identical block
            ident_sig = sig_mgr.get_identical_block_signature(row_i)
            if ident_sig is not None:
                # row i must be identical to other block members
                v = utils.bitset_to_vector(ident_sig, k)
                if utils.weighted_i_compatible(A, B, W, v, row_i):
                    # no need to add this signature
                    B[row_i] = v
                    filled = True
            else:
                # check row_i's constraints
                forbidden = np.zeros(k, dtype=bool)

                for u in range(row_i):
                    if A[row_i][u] == 0:
                        # there is no edge between these vertices, so they cannot share any cliques
                        forbidden = np.logical_or(forbidden, B[u])

                # create candidate pattern
                pattern = utils.PatternIterator(forbidden)
                while True:
                    v = pattern.next()
                    if v is None:
                        break

                    # test compatibility
                    sig = pattern.get_int()
                    if sig_mgr.is_valid(row_i, sig) and utils.weighted_i_compatible(A, B, W, v, row_i):
                        B[row_i] = v
                        sig_mgr.add_signature(row_i, sig)
                        filled=True
                        break

            if not filled:
                return B, row_i
        row_i+=1

    return B, A.shape[0] 


def extend_basis_v3(A, B, W, k):
    """
    PT1: enabled
    PT2: disabled
    """
    row_i = 0

    # Note: do not use enumerate() for better performance
    for Bi in B:
        filled = False
        
        if np.all(Bi == -1):
            # check row_i's constraints
            forbidden = np.zeros(k, dtype=bool)

            for u in range(row_i):
                if A[row_i][u] == 0:
                    # there is no edge between these vertices, so they cannot share any cliques
                    forbidden = np.logical_or(forbidden, B[u])

            # create candidate pattern
            pattern = utils.PatternIterator(forbidden)
            while True:
                v = pattern.next()
                if v is None:
                    break

                # test compatibility
                if utils.weighted_i_compatible(A, B, W, v, row_i):
                    B[row_i] = v
                    filled=True
                    break

            if not filled:
                return B, row_i
        row_i+=1

    return B, A.shape[0] 


def get_icomp_basis_row(A, B, W, book, model, dv_w, perf_lp_v2_enabled):
    """
    Returns: tuple of (next basis row, number of LP runs in this function call)
    """
    # Gets the next basis row that is i-compatible.
    icomp=False
    num_lp_runs = 0

    while not icomp:  # get basis row until icomp
        if perf_lp_v2_enabled:
            Pb, sig = book.get_next_patt_row(A)
        else:
            Pb = book.get_next_patt_row()
        
        if Pb[0] == -1:
            status = book.backtrack(B, model)
            if status is False:
                return [-1], num_lp_runs
            
            W = book.Ws[book.b] # resets W to the W before inserting current book.b 
        else:
            count=0         
            count = add_model_constraints(A, B, Pb, book, dv_w, model)
            feasible = optimize_model(model, book, W)  # updates W if feasible
            num_lp_runs += 1
                
            if not feasible:
                utils.remove_constraints(model, count)
                if perf_lp_v2_enabled:
                    book.sig_mgr.remove_signature(book.i, sig)
                continue
            
            icomp=True
            book.cnstrs_added[book.b]+=count
            book.basis_inds[book.b] = book.i
            
    return Pb, num_lp_runs  # returning icomp basis row 
    
    
def BSWD_DW(A, k, perf_lp_v2_enabled, perf_lp_v3_enabled, num_threads):
    """
    Returns: triple of (basis matrix B, weight matrix W, number of LP runs)
    """
    perf_lp_v2_enabled |= perf_lp_v3_enabled
    n=A.shape[0]
    
    if 2*k>n:
        max_val = n
    else:
        max_val = 2*k
    
    if perf_lp_v2_enabled:
        #---------- Compute block information
        block_ids = utils.compute_blocks(A)
        book = BookeeperV2(k, max_val, block_ids)
    else:
        book = Bookeeper(k, max_val)

    #---------- Create a new model
    num_lp_runs = 0
    model = gp.Model("weights")
    if num_threads is not None:
        model.setParam(GRB.Param.Threads, num_threads)  # number of threads for Gurobi
    model.Params.LogToConsole = 0  # dont print to console
    
    #---------- Add Decision Variables [w0, w1, ..., wk] 1-D array
    dv_w = model.addMVar(shape=(k), vtype=GRB.INTEGER, name="w")
    model.addConstr(dv_w >= 0)  #non-negativity constraint

    #---------- Set objective function
    model.setObjective(dv_w.sum(), GRB.MINIMIZE)
    model.update()
    
    W = utils.construct_weight_matrix([1]*k, [0]*k, k) # init weight matrix

    B_tilde = np.full((n, k), -1)   # nxk 'null' rows
    backtracked=False    
    
    # book.b: our current basis row [0, 1, ..., maxval-1]
    # book.i: the index of B we are trying to insert basis row
    while book.b < max_val: 
        if backtracked:
            book.b-=1
            
        book.Ws[book.b] = np.copy(W)  # W before inserting book.b basis row
        
        Pb, num_lp_runs_part = get_icomp_basis_row(A, B_tilde, W, book, model, dv_w, perf_lp_v2_enabled)
        num_lp_runs += num_lp_runs_part
            
        if Pb[0]==-1: # failed
            return np.full((1,1), -1), np.full((1,1), -1), num_lp_runs  # bsd failed
        
        B_tilde[book.i] = Pb  

        B_tilde_copy = np.copy(B_tilde) 
        
        if perf_lp_v3_enabled:
            B, i = extend_basis_v3(A, B_tilde_copy, W, k)
        elif perf_lp_v2_enabled:
            B, i = extend_basis_v2(A, B_tilde_copy, W, k, book.sig_mgr.copy())
        else:
            B, i = extend_basis(A, B_tilde_copy, W, book.cart_V, k)
        book.i = i
        
        if book.i == n:
            return B, W, num_lp_runs  # bsd found soln
        
        # backtracking not finished yet but while loop is about to quit
        backtracked=False
        if book.b+1 == max_val and not book.is_completed():
            book.i = book.basis_inds[book.b]
            book.backtrack(B_tilde, model)
            backtracked=True
            
        book.b+=1
        
    return np.full((1,1), -1), np.full((1,1), -1), num_lp_runs  # bsd failed


            
   
