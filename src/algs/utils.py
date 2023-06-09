
import pandas as pd
import numpy as np
import sympy, os
from collections import Counter
import itertools, random, math, re
from collections import deque 
from typing import Optional, List

   
def equiv_rows(u_row, a_row):
    '''
    Args: u_row : 1-D pandas array
          a_row : 1-D pandas array
    '''
    for i, val in u_row.items():
        if not equiv_ints(u_row[i], a_row[i]):
            return False
    return True


def equiv_ints(u, a):
    if u != a and a != math.inf and u != math.inf:  
        return False
    else:
        return True
        

def w_limited_guess(cart, *args, numrows=1, numcols=1, index=0, w=0):
    # used in non-backtracking versions
    P = np.array(cart.entryAt(index)).reshape(numrows,numcols)
    
    # check if compatible (w-limitness)
    compat = P.dot(P.T).max()
    
    # check for lin independence
    # reduced row echelon form
    # inds are indices of pivot cols 
    _, inds = sympy.Matrix(P).T.rref() 
    
    if compat <= w and len(inds)==len(P):
        return P
    else:
        return np.full((1,1), -1)  
    

def weighted_i_compatible(A, B, W, v, i):    
    aii = A[i, i]
    
    vW = np.dot(v, W)
    vWvT = np.dot(vW, v.T)
    
    # first condition
    if not equiv_ints(vWvT, aii):
        return False
        
    # second condition
    j = 0
    compat = True
    for Bj in B:
        if j != i and not np.all(Bj==-1):  # i!=j and Bj not null row
            Aij = A[i, j]
            vWBjT = np.dot(vW, Bj.T)
            if vWBjT != Aij:
                compat = False
        j+=1
    return compat


def construct_weight_matrix(v, vals, k):
    '''
    v: the diagonal indices to actually insert values
    vals: the weight values
    
    Inserts values into the diagonal weight matrix 
    '''
    W = np.zeros((k, k))
    
    fillcount=0
    for i in range(W.shape[0]):  # makes diagonal null values
        if v[i] == 0:
            W[i, i] = np.inf
        else:
            if vals[fillcount] == -0.0:
                W[i,i] = 0.0
            else: 
                W[i,i] = vals[fillcount]
            fillcount+=1
    return W


def is_lin_indep(P, Pb):    
    # check for lin indep w. previous rows, returns T/F + rank
    P.append(Pb)
    _, inds = sympy.Matrix(P).T.rref() 
    psize = len(P)
    P.pop()
    
    if len(inds)==psize:
        return True, len(inds)
    else:
        return False, len(inds)
    

def remove_constraints(model, count):
    # removes the last count number of constraints added to model
    for c in range(count):
        constraint = model.getConstrs()[-1-c]
        model.remove(constraint)
    model.update()


#----------------------------------------------------- ip find weights
class Weights:
    def __init__(self, k):
        self.W_0 = construct_weight_matrix([1]*k, [-1]*k, k) # init weight matrix
        self.w_deque = deque()
        self.w_deque.append(self.W_0)
        
        
    def extend(self, ws):
        self.w_deque.extendleft(ws)
    
    def appendleft(self, w):
        self.w_deque.appendleft(w)
    
    def print_ws(self):
        for w in self.w_deque:
            print(w)

def get_one_rnd_perm(k_total, k_distinct):
    # used in datagen.py
    # finds combs sum to A, then finds all permutations of each comb 
    if k_distinct==1:
        return [k_total] 
    
    combos = combo(k_total, k_distinct)
    
    valid_combos = []
    for comb in combos:
        if len(list(comb.elements())) == k_distinct:
            valid_combos.append(comb)
    
    number_potential = len(valid_combos)
    
    rnd_perm_ind = random.randint(0, number_potential-1) 
    
    clique_weights = list(valid_combos[rnd_perm_ind].elements())
        
    random.shuffle(clique_weights)
    
    return clique_weights


def combo(total, n):
    '''
    Find all combinations of n integers from 0 to total that sum to total
    '''
    weights = range(0,total+1)
    
    # initialize list of list for potential weights
    pws = [[Counter()]] + [[] for _ in range(total)]   # A/maxweight+1 time
    
    # this is pretty close to being O(n^2)
    for weight in weights:  # {0, 1, 2, 3, ..., total} A/maxweight time
        for i in range(weight, total + 1): # A/maxweight+1 time 
                        
            # increment pws at index i and add a weight
            for pw in pws[i-weight]:
                
                # prevents combos w. too many elements
                if len(list(pw.elements()))+1<=n:          
                    pws[i] += [pw + Counter({weight: 1})]
             
    return pws[total]


def get_combos(sum_to, num_ints):
    if num_ints==1:
        return [Counter({int(sum_to) : 1})]
    
    combos = combo(int(sum_to), int(num_ints))
    return combos


def update(Aij, W, v, Bj, w_deque):
    '''
    Returns list of new weights that are compatible
    '''    
    vW = np.dot(v, W)
    vWBjT = np.dot(vW, Bj.T)
    
    # gets the number of weights to find at this iteration
    # also the sum given the past weights
    num_ints=0
    ind=0
    sum_to = Aij
    
    for l in vW:
        if l==-1 and Bj[ind]==1:
            num_ints+=1
        elif l > 0 and Bj[ind] == 1:
            sum_to-=l     
        ind+=1
   
    # only find permuations when we need to 
    if sum_to > 0 and num_ints > 0:
        combs = get_combos(sum_to, num_ints)

        # --Now, update W
        for comb in combs:
            combination = list(comb.elements()) 

            # specific combo has proper # of ints
            if len(combination) == num_ints: 
            
                W_copy = np.copy(W) 
                numfilled = 0
                v_ind = 0
                
                # insert new weights in correct pos. in W
                for v_val in v:             
                    if vW[v_ind] == -1 and v_val == 1 and Bj[v_ind]==1:
                        W_copy[v_ind, v_ind] = combination[numfilled]
                        numfilled+=1
                    v_ind+=1
                    
                w_deque.appendleft(W_copy)
    else:
        # if we dont to update weights, checks for compatiblity
        # this check is necessary
        if equiv_ints(vWBjT, Aij):
            w_deque.appendleft(W)


class PatternRows:
    '''
    If the full cartesian product is a set A, then this function 
    generates A[index], rather than creating all values of the 
    cartesian product at once.
    '''
    def __init__(self, k):
        self.sets = [0,1]
        self.divs = []
        self.mod = 2 # size of each list = 2
        self.k = k
        self.maxSize = 2 ** self.k
        self.init()

    def init(self):
        length = self.k
        factor = 1
        for i in range((length - 1), -1, -1):
            items = len(self.sets)
            self.divs.insert(0, factor)
            factor = factor * items
                    
    def entryAt(self, n):
        length = self.k
        if n < 0 or n >= self.maxSize:
            raise IndexError
        combination = []
        for i in range(0, length):
            combination.append(self.sets[ int(math.floor(n / self.divs[i])) % self.mod])
        return combination


class PatternIterator:
    """
    TODO: add documentation
    """
    def __init__(self, forbidden: np.ndarray):
        n = forbidden.shape[0]
        self._v = np.zeros(n)

        # indices of the available bits; we should not keep numpy int but Python one to handle large integers
        self._avail = [i for i, x in enumerate(forbidden) if not x]
        self._cursor = 0  # keeps track of the number of leading ones among the available positions
        self._int_val = 0

    def __len__(self) -> int:
        """TODO: add documentation"""
        return self._v.shape[0]

    def get(self) -> np.ndarray:
        """TODO: add documentation"""
        return self._v

    def get_int(self) -> int:
        """
        Returns the integer representation of the current pattern.

        The i-th bit in the pattern vector corresponds to the i-th bit of
        the resulting integer.
        
        (e.g. vector [1, 0, 1, 0] corresponds to 1*(2**0) + 1*(2**2) = 5)
        """
        return self._int_val

    def has_next(self) -> bool:
        """TODO: add documentation"""
        return self._cursor < len(self._avail)

    def next(self) -> Optional[np.ndarray]:
        """TODO: add documentation"""

        if not self.has_next():
            return None

        index = self._avail[self._cursor]
        self._v[index] = 1  # set this bit
        self._int_val |= 1 << index

        if self._cursor == 0:
            self._cursor += 1

            # move the cursor to where the first zero occurs
            while self.has_next() and self._v[self._avail[self._cursor]] != 0:
                self._cursor += 1
        else:
            self._v[:index] = 0  # reset lower bits
            self._cursor = 0  # reset cursor

            # Reset lower bits by bit shifting.
            #
            # x >> y << y is equivalent to (x >> y) << y,
            # shifting y bits to the right and then y bits to the left.
            # For any non-negative integer x (of unbounded bit length in Python),
            # x >> y gives (floor(x / (2^y))) and
            # (x >> y) << y gives (2^y * floor(x / (2^y))), which clears the lowest y bits in x.
            # example: 0b111 >> 2 == 0b1; 0b111 >> 2 << 2 = 0b100
            self._int_val = self._int_val >> index << index

        return self._v


def compute_blocks(A: np.ndarray) -> List[int]:
    """
    Returns block information for the given adjacency matrix.

    TODO: merge algs.kernel.num_blocks() (which accepts pandas.DataFrame) with this
    """

    n = A.shape[0]
    SINGLETON = -1
    block_ids = [SINGLETON] * n  # block_ids[i] := vertex i's block id
    next_block_id = 0

    for i in range(n):
        if block_ids[i] >= 0:
            continue  # already assigned

        # try to find all of i's equiv rows
        has_equiv_row = False
        for j in range(i + 1, n):
            if A[i][j] > 0 and all(equiv_ints(A[i][x], A[j][x]) for x in range(n)):
                has_equiv_row = True
                block_ids[j] = next_block_id

        # update if this block has more than one members
        if has_equiv_row:
            block_ids[i] = next_block_id
            next_block_id += 1

    return block_ids

def bitset_to_vector(x: int, k: int) -> np.ndarray:
    """
    Converts an integer as a bitset to the corresponding Numpy array of binary elements.
    The i-the element of the resulting array is 1 if and only if the i-th bit of x is on.

    Time complexity: O(k)
    """
    return np.array([(x >> i) & 1 for i in range(k)])
