from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce

import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import eigs as sps_eigs
from mpo import *

def graphic(sys_block, env_block, sys_label="r"):
    if sys_label == "l":
        graphic = ("=" * env_block) + "*" + ("-" * sys_block)
    else:
        graphic = ("=" * sys_block) + "*" + ("-" * env_block)

    return graphic

class MatrixState(object):
    """
    Matrix state for a single site. A 3-degree tensor with 2 bond degrees to other matrix states and a physical degree.
    A matrix product operator (MPO) is also included in the matrix state.
    A sentinel matrix state could be initialized for an imaginary state
    which provides convenience for doubly linked list implementation.
    """

    def __init__(self, bond_dim1, bond_dim2, mpo, error_thresh=0, sitenum=0, Totsite=50):
        """
        Initialize a matrix state with shape (bond_dim1, phys_d, bond_dim2) and an MPO attached to the state,
        where phys_d is determined by the MPO and MPO is usually the Hamiltonian.
        If a sentinel `MatrixState` is required, set bond_dim1, phys_d or bond_dim2 to 0 or None.
        MPO should be a 4-degree tensor with 2 bond degrees at first and 2 physical degrees at last.
        :parameter bond_dim1: shape[0] of the matrix
        :parameter bond_dim2: shape[2] of the matrix
        :parameter mpo: matrix product operator (hamiltonian) attached to the matrix
        :parameter error_thresh: error threshold used in svd compressing of the matrix state.
        The lower the threshold, the higher the accuracy.
        """
        # if is a sentinel matrix state
        if not (bond_dim1 and bond_dim2):
            self._matrix = self.mpo = np.ones((0, 0, 0))
            self.left_ms = self.right_ms = None
            self.F_cache = self.L_cache = self.R_cache = np.ones((1,) * 6)
#            self.FCI_cache = np.ones((1,) * 4)
            self.is_sentinel = True
            return
        self.is_sentinel = False
        phys_d = mpo.shape[2]
        # random initialization of the state tensor
         
        bond_dim1 = min( bond_dim1, phys_d ** ( sitenum - 1 ), phys_d ** ( Totsite - sitenum + 1 ) )
        bond_dim2 = min( bond_dim2, phys_d ** sitenum, phys_d ** ( Totsite - sitenum ) )
        print(sitenum,bond_dim1,phys_d,bond_dim2)
#        self.bond_dim1 = bond_dim1 
#        self.bond_dim2 = bond_dim2 
#        self.phys_d = phys_d 

        self._matrix = np.random.random((bond_dim1, phys_d, bond_dim2))
#        self._matrix = np.zeros((bond_dim1, phys_d, bond_dim2))

        self.mpo = mpo 
        self.mpo_parity = mpo 
        # the pointer to the matrix state on the left
        self.left_ms = None
        # the pointer to the matrix state on the right
        self.right_ms = None
        # cache for F, L and R to accelerate calculations
        # for the definition of these parameters, see the [reference]: Annals of Physics, 326 (2011), 145-146
        # because of the cache, any modifications to self._matrix should be properly wrapped.
        # modifying self._matrix directly may lead to unexpected results.
        self.F_cache = None
        self.L_cache = self.R_cache = None
#        self.FCI_cache = None
        self.error_thresh = error_thresh
        self.site_num = sitenum
        self.Totsite_num = Totsite

    @classmethod
    def create_sentinel(cls):
        return cls(0, 0, None)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix):
        # bond dimension may have reduced due to low local degree of freedom
        # but the dimension of the physical degree must not change
        assert self.phys_d == new_matrix.shape[1]
        self._matrix = new_matrix
        # forbid writing for safety concerns
        self._matrix.flags.writeable = False
        # disable the cache for F, L, R
        self.clear_cache()

    @property
    def bond_dim1(self):
        """
        :return: the dimension of the first bond degree
        """
        return self.matrix.shape[0]

    @property
    def phys_d(self):
        """
        :return: the dimension of the physical index
        """
        assert self.matrix.shape[1] == self.mpo.shape[2] == self.mpo.shape[3]
        return self.matrix.shape[1]

    @property
    def bond_dim2(self):
        """
        :return: the dimension of the second bond degree
        """
        return self.matrix.shape[2]

    def svd_compress(self, direction):
        """
        Perform svd compression on the self.matrix. Used in the canonical process.
        :param direction: To which the matrix is compressed
        :return: The u,s,v value of the svd decomposition. Truncated if self.thresh is provided.
        """
        left_argument_set = ["l", "left"]
        right_argument_set = ["r", "right"]
        assert direction in (left_argument_set + right_argument_set)
        if direction in left_argument_set:
            u, s, v = svd(
                self.matrix.reshape(self.bond_dim1 * self.phys_d, self.bond_dim2),
                full_matrices=False,
            )
        else:
            u, s, v = svd(
                self.matrix.reshape(self.bond_dim1, self.phys_d * self.bond_dim2),
                full_matrices=False,
            )
        if self.error_thresh == 0:
            return u, s, v
        new_bond_dim = max(
            ((s.cumsum() / s.sum()) < 1 - self.error_thresh).sum() + 1, 1
        )
        return u[:, :new_bond_dim], s[:new_bond_dim], v[:new_bond_dim, :]

    def left_canonicalize(self):
        """
        Perform left canonical decomposition on this site
        """
        if not self.right_ms:
            return
        u, s, v = self.svd_compress("left")
        self.matrix = u.reshape((self.bond_dim1, self.phys_d, -1))
        self.right_ms.matrix = np.tensordot(
            np.dot(np.diag(s), v), self.right_ms.matrix, axes=[1, 0]
        )

    def left_canonicalize_all(self):
        """
        Perform left canonical decomposition on this site and all sites on the right
        """
        if not self.right_ms:
            return
        self.left_canonicalize()
        self.right_ms.left_canonicalize_all()

    def right_canonicalize(self):
        """
        Perform right canonical decomposition on this site
        """
        if not self.left_ms:
            return
        u, s, v = self.svd_compress("right")
        self.matrix = v.reshape((-1, self.phys_d, self.bond_dim2))
        self.left_ms.matrix = np.tensordot(
            self.left_ms.matrix, np.dot(u, np.diag(s)), axes=[2, 0]
        )

    def right_canonicalize_all(self):
        """
        Perform right canonical decomposition on this site and all sites on the left
        """
        if not self.left_ms:
            return
        self.right_canonicalize()
        self.left_ms.right_canonicalize_all()

    def test_left_unitary(self):
        """
        Helper function to test if this site is left normalized
        Only for test. Not used in release version
        """
        m = self.matrix
        summation = sum(
            [
                np.dot(m[:, i, :].transpose().conj(), m[:, i, :])
                for i in range(self.phys_d)
            ]
        )
        print(
            "Test left unitary: %s" % np.allclose(summation, np.eye(self.bond_dim2))
        )

    def test_right_unitary(self):
        """
        Helper function to test if this site is right normalized
        Only for test. Not used in release version
        """
        m = self.matrix
        summation = sum(
            [
                np.dot(m[:, i, :], m[:, i, :].transpose().conj())
                for i in range(self.phys_d)
            ]
        )
        print(
            "Test right unitary: %s" % np.allclose(summation, np.eye(self.bond_dim1))
        )

    def calc_F(self, mpo=None):
        """
        calculate F for this site.
        graphical representation (* for MPS and # for MPO,
        numbers represents a set of imaginary bond dimensions used for comments below):
        :parameter mpo: an external MPO to calculate. Used in expectation calculation.
        :return the calculated F
        """
        # whether use self.mpo or external MPO
        use_self_mpo = mpo is None
        if use_self_mpo:
            mpo = self.mpo
        # return cache immediately if the value has been calculated before and self.matrix has never changed
#        if use_self_mpo and self.F_cache is not None:
#            return self.F_cache
        # Do the contraction from top to bottom.
        # suppose self.matrix.shape = 1,4,5, self.mpo.shape = 2,3,4,4 (left, right, up, down)
        """
          0 --*-- 2                   0 --*-- 1                   
              |                           |                       0 --*-- 1
              1                       2 --#-- 3                       |    
                     --tensordot-->       |       --tensordot-->  2 --#-- 3
              2                           4                           |    
              |                                                   4 --#-- 5  
          0 --#-- 1                       1                       
              |                           |                       
              3                       0 --#-- 2                   
        """
        # up_middle is of shape (1, 5, 2, 3, 4)
#        up_middle = np.tensordot(self.matrix.conj(), mpo, axes=[1, 2])
        up_middle = np.tensordot(self.matrix, mpo, axes=[1, 2])
        # return value F is of shape (1, 5, 2, 3, 1, 5). In the graphical representation,
        # the position of the degrees of the tensor is from top to bottom and left to right
        F = np.tensordot(up_middle, self.matrix, axes=[4, 1])
        if use_self_mpo:
            pass
            self.F_cache = F
        return F

    def calc_F_parity(self, mpo=None):
        """
        calculate F for this site.
        graphical representation (* for MPS and # for MPO,
        numbers represents a set of imaginary bond dimensions used for comments below):
        :parameter mpo: an external MPO to calculate. Used in expectation calculation.
        :return the calculated F
        """
        # whether use self.mpo or external MPO
        use_self_mpo = mpo is None
        if use_self_mpo:
            mpo = self.mpo_parity
        # return cache immediately if the value has been calculated before and self.matrix has never changed
#        if use_self_mpo and self.F_cache is not None:
#            return self.F_cache
        # Do the contraction from top to bottom.
        # suppose self.matrix.shape = 1,4,5, self.mpo.shape = 2,3,4,4 (left, right, up, down)
        """
          0 --*-- 2                   0 --*-- 1                   
              |                           |                       0 --*-- 1
              1                       2 --#-- 3                       |    
                     --tensordot-->       |       --tensordot-->  2 --#-- 3
              2                           4                           |    
              |                                                   4 --#-- 5  
          0 --#-- 1                       1                       
              |                           |                       
              3                       0 --#-- 2                   
        """
        # up_middle is of shape (1, 5, 2, 3, 4)
#        up_middle = np.tensordot(self.matrix.conj(), mpo, axes=[1, 2])
        up_middle = np.tensordot(self.matrix, mpo, axes=[1, 2])
        # return value F is of shape (1, 5, 2, 3, 1, 5). In the graphical representation,
        # the position of the degrees of the tensor is from top to bottom and left to right
        F = np.tensordot(up_middle, self.matrix, axes=[4, 1])
        if use_self_mpo:
            pass
            self.F_cache = F
        return F

    def calc_L_static(self, OBJ):
        """
        calculate L in a static way
        """
        isite = self.site_num
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1          0 --*-- 1                   0 --*-- 3                   0 --*-- 1          
              |                  |                           |                           |     
          2 --#-- 3     +    2 --#-- 3  --tensordot-->   1 --#-- 4    --reshape-->   2 --#-- 3                
              |                  |                           |                           |     
          4 --*-- 5          4 --*-- 5                   2 --*-- 5                   4 --*-- 5          
        
        """
        L = OBJ[0].calc_F_parity()
        for i in range(isite-1):
            Ftmp = OBJ[i+1].calc_F_parity() 
#            print("L shape=",L.shape)
#            print("F shape=",Ftmp.shape)
            L = np.tensordot(L, Ftmp, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
#        self.L_cache = L
        return L

    def calc_L(self):
        """
        calculate L in a recursive way
        """
        # the left state is a sentinel, return F directly.
        if not self.left_ms:
            return self.calc_F()
        # return cache immediately if available
        if self.L_cache is not None:
            return self.L_cache
        # find L from the state on the left
        last_L = self.left_ms.calc_L()
        # calculate F in this state
        F = self.calc_F()
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1          0 --*-- 1                   0 --*-- 3                   0 --*-- 1          
              |                  |                           |                           |     
          2 --#-- 3     +    2 --#-- 3  --tensordot-->   1 --#-- 4    --reshape-->   2 --#-- 3                
              |                  |                           |                           |     
          4 --*-- 5          4 --*-- 5                   2 --*-- 5                   4 --*-- 5          
        
        """
        L = np.tensordot(last_L, F, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
        self.L_cache = L
        return L

    def calc_L_parity(self):
        """
        calculate L in a recursive way
        """
        # the left state is a sentinel, return F directly.
        if not self.left_ms:
            return self.calc_F_parity()
        # return cache immediately if available
        if self.L_cache is not None:
            return self.L_cache
        # find L from the state on the left
        last_L = self.left_ms.calc_L_parity()
        # calculate F in this state
        F = self.calc_F_parity()
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1          0 --*-- 1                   0 --*-- 3                   0 --*-- 1          
              |                  |                           |                           |     
          2 --#-- 3     +    2 --#-- 3  --tensordot-->   1 --#-- 4    --reshape-->   2 --#-- 3                
              |                  |                           |                           |     
          4 --*-- 5          4 --*-- 5                   2 --*-- 5                   4 --*-- 5          
        
        """
        L = np.tensordot(last_L, F, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
        self.L_cache = L
        return L

    def calc_R_static(self, OBJ):
        """
        calculate R in a static way
        """
        isite = self.site_num
        tot_num = self.Totsite_num

        R = OBJ[tot_num-1].calc_F_parity()
        for i in range(tot_num - isite):
            Ftmp = OBJ[tot_num - i - 2].calc_F_parity() 
            R = np.tensordot(Ftmp, R, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
#        self.R_cache = R
        return R

    def calc_R(self):
        """
        calculate R in a recursive way
        """
        # mirror to self.calc_L. Explanation omitted.
        if self.R_cache is not None:
            return self.R_cache
        if not self.right_ms:
            return self.calc_F()
        last_R = self.right_ms.calc_R()
        F = self.calc_F()
        R = np.tensordot(F, last_R, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
        self.R_cache = R
        return R

    def calc_R_parity(self):
        """
        calculate R in a recursive way
        """
        # mirror to self.calc_L. Explanation omitted.
        if self.R_cache is not None:
            return self.R_cache
        if not self.right_ms:
            return self.calc_F_parity()
        last_R = self.right_ms.calc_R_parity()
        F = self.calc_F_parity()
        R = np.tensordot(F, last_R, axes=[[1, 3, 5], [0, 2, 4]]).transpose(
            (0, 3, 1, 4, 2, 5)
        )
        self.R_cache = R
        return R

    def calc_FCI_H(self):
        """
        calculate Full CI Hamiltonian in a recursive way
        """
        # the left state is a sentinel, return F directly.
        if not self.left_ms:
            return self.calc_FCI_H()
        # return cache immediately if available
        if self.FCI_cache is not None:
            return self.FCI_cache
        # find L from the state on the left
        last_FCI_H = self.left_ms.calc_FCI_H()

        mpo = self.mpo
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         2     4    
              |                  |                           |     |                         |     |    
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#-- 1  
              |                  |                           |     |                         |     |    
              3                  3                           2     5                         3     5    
        
        """
        L = np.tensordot(last_FCI, mpo, axes=[[1], [0]]).transpose(
            (0, 3, 1, 2, 4, 5)
        )
        self.L_cache = L
        return L

    def clear_cache(self):
        """
        clear cache for F, L and R when self.matrix has changed
        """
        self.F_cache = None
        self.FCI_cache = None
        # clear R cache for all matrix state on the left because their R involves self.matrix
        self.left_ms.clear_R_cache()
        # clear L cache for all matrix state on the right because their L involves self.matrix
        self.right_ms.clear_L_cache()

    def clear_L_cache(self):
        """
        clear all cache for L in matrix states on the right in a recursive way
        """
        # stop recursion if the end of the MPS is met
        if self.L_cache is None or not self:
            return
        self.L_cache = None
        self.right_ms.clear_L_cache()

    def clear_R_cache(self):
        """
        clear all cache for R in matrix states on the left in a recursive way
        """
        # stop recursion if the end of the MPS is met
        if self.R_cache is None or not self:
            return
        self.R_cache = None
        self.left_ms.clear_R_cache()

    def calc_variational_tensor_static(self, OBJ):
        """
        calculate the variational tensor for the ground state search. L * MPO * R
        graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
        """
        Lend = not self.left_ms
        Rend = not self.right_ms
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1                                    0 --*-- 1                
              |                | 2                         |    | 6      
          2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5                 
              |                | 3                         |    | 7      
          4 --*-- 5                                    3 --*-- 4                
              L                MPO                       left_middle
        """
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1             0 --*-- 1                   0 --*-- 1 7 --*-- 8      
              |    | 6              |                           |    | 5  |   
          2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 9       
              |    | 7              |                           |    | 6  |   
          3 --*-- 4             4 --*-- 5                   3 --*-- 4 10--*-- 11 
            left_middle             R                       raw variational tensor
        Note the dimension of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
        """
        if not Rend and not Lend:

#            print(self.left_ms.calc_L_static(OBJ).shape)
#            print(self.mpo.shape)
             
            left_middle = np.tensordot(self.left_ms.calc_L_static(OBJ), self.mpo_parity, axes=[3, 0])
            raw_variational_tensor = np.tensordot(
                left_middle, self.right_ms.calc_R_static(OBJ), axes=[5, 2]
            )

#            #test energy each site
#            Etmp = np.tensordot(raw_variational_tensor, self.matrix, axes=[[4, 6, 10], [0, 1, 2]])
#            E    = np.tensordot(Etmp , self.matrix, axes=[[1, 4, 5], [0, 1, 2]])
#            print(E)

            shape = (
                self.bond_dim1,
                self.bond_dim1,
                self.phys_d,
                self.phys_d,
                self.bond_dim2,
                self.bond_dim2,
            )
            H_eff = raw_variational_tensor.reshape(shape).transpose((0, 2, 4, 1, 3, 5))
            """
              *-- 0 4 --*
              |    | 2  |
              #----#----#
              |    | 3  |
              *-- 1 5 --*
            """
        else: 
            if Rend: 
                left_middle = np.tensordot(self.left_ms.calc_L_static(OBJ), self.mpo_parity, axes=[3, 0])

#                #test energy each site
#                Etmp = np.tensordot(left_middle, self.matrix, axes=[[4, 7], [0, 1]])
#                E    = np.tensordot(Etmp , self.matrix, axes=[[1, 5], [0, 1]])
#                print(E)

                shape = (
                    self.left_ms.bond_dim2,
                    self.left_ms.bond_dim2,
                    self.phys_d,
                    self.phys_d,
                )
                """
                  *-- 0  
                  |    | 2     
                  #----#
                  |    | 3     
                  *-- 1 
                """
                H_eff = left_middle.reshape(shape).transpose((0, 2, 1, 3))

            else:
                raw_variational_tensor = np.tensordot(
                    self.mpo_parity, self.right_ms.calc_R_static(OBJ), axes=[1, 2]
                )

#                #test energy each site
#                Etmp = np.tensordot(raw_variational_tensor, self.matrix, axes=[[2, 6], [1, 2]])
#                E    = np.tensordot(Etmp , self.matrix, axes=[[1, 2], [1, 2]])
#                print(E)

                shape = (
                    self.phys_d,
                    self.phys_d,
                    self.right_ms.bond_dim1,
                    self.right_ms.bond_dim1,
                )
                """
                   2 --*
                0 |    |
                  #----#
                1 |    |
                   3 --*
                """
                H_eff = raw_variational_tensor.reshape(shape).transpose((0, 2, 1, 3))

        return H_eff

    def calc_variational_tensor(self):
        """
        calculate the variational tensor for the ground state search. L * MPO * R
        graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
        """
        Lend = not self.left_ms
        Rend = not self.right_ms
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1                                    0 --*-- 1                
              |                | 2                         |    | 6      
          2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5                 
              |                | 3                         |    | 7      
          4 --*-- 5                                    3 --*-- 4                
              L                MPO                       left_middle
        """
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1             0 --*-- 1                   0 --*-- 1 7 --*-- 8      
              |    | 6              |                           |    | 5  |   
          2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 9       
              |    | 7              |                           |    | 6  |   
          3 --*-- 4             4 --*-- 5                   3 --*-- 4 10--*-- 11 
            left_middle             R                       raw variational tensor
        Note the dimension of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
        """
        if not Rend and not Lend:

            left_middle = np.tensordot(self.left_ms.calc_L(), self.mpo, axes=[3, 0])
            raw_variational_tensor = np.tensordot(
                left_middle, self.right_ms.calc_R(), axes=[5, 2]
            )

#            #test energy each site
#            Etmp = np.tensordot(raw_variational_tensor, self.matrix, axes=[[4, 6, 10], [0, 1, 2]])
#            E    = np.tensordot(Etmp , self.matrix, axes=[[1, 4, 5], [0, 1, 2]])
#            print(E)

            shape = (
                self.bond_dim1,
                self.bond_dim1,
                self.phys_d,
                self.phys_d,
                self.bond_dim2,
                self.bond_dim2,
            )
            H_eff = raw_variational_tensor.reshape(shape).transpose((0, 2, 4, 1, 3, 5))
            """
              *-- 0 4 --*
              |    | 2  |
              #----#----#
              |    | 3  |
              *-- 1 5 --*
            """
        else: 
            if Rend: 
                left_middle = np.tensordot(self.left_ms.calc_L(), self.mpo, axes=[3, 0])

#                #test energy each site
#                Etmp = np.tensordot(left_middle, self.matrix, axes=[[4, 7], [0, 1]])
#                E    = np.tensordot(Etmp , self.matrix, axes=[[1, 5], [0, 1]])
#                print(E)

                shape = (
                    self.left_ms.bond_dim2,
                    self.left_ms.bond_dim2,
                    self.phys_d,
                    self.phys_d,
                )
                """
                  *-- 0  
                  |    | 2     
                  #----#
                  |    | 3     
                  *-- 1 
                """
                H_eff = left_middle.reshape(shape).transpose((0, 2, 1, 3))

            else:
                raw_variational_tensor = np.tensordot(
                    self.mpo, self.right_ms.calc_R(), axes=[1, 2]
                )

#                #test energy each site
#                Etmp = np.tensordot(raw_variational_tensor, self.matrix, axes=[[2, 6], [1, 2]])
#                E    = np.tensordot(Etmp , self.matrix, axes=[[1, 2], [1, 2]])
#                print(E)

                shape = (
                    self.phys_d,
                    self.phys_d,
                    self.right_ms.bond_dim1,
                    self.right_ms.bond_dim1,
                )
                """
                   2 --*
                0 |    |
                  #----#
                1 |    |
                   3 --*
                """
                H_eff = raw_variational_tensor.reshape(shape).transpose((0, 2, 1, 3))

        return H_eff

    def calc_variational_tensor_parity(self):
        """
        calculate the variational tensor for the ground state search. L * MPO * R
        graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
        """
        Lend = not self.left_ms
        Rend = not self.right_ms
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1                                    0 --*-- 1                
              |                | 2                         |    | 6      
          2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5                 
              |                | 3                         |    | 7      
          4 --*-- 5                                    3 --*-- 4                
              L                MPO                       left_middle
        """
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1             0 --*-- 1                   0 --*-- 1 7 --*-- 8      
              |    | 6              |                           |    | 5  |   
          2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 9       
              |    | 7              |                           |    | 6  |   
          3 --*-- 4             4 --*-- 5                   3 --*-- 4 10--*-- 11 
            left_middle             R                       raw variational tensor
        Note the dimension of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
        """
        if not Rend and not Lend:
            left_middle = np.tensordot(self.left_ms.calc_L_parity(), self.mpo_parity, axes=[3, 0])
            raw_variational_tensor = np.tensordot(
                left_middle, self.right_ms.calc_R_parity(), axes=[5, 2]
            )

#            #test energy each site
#            Etmp = np.tensordot(raw_variational_tensor, self.matrix, axes=[[4, 6, 10], [0, 1, 2]])
#            E    = np.tensordot(Etmp , self.matrix, axes=[[1, 4, 5], [0, 1, 2]])
#            print(E)

            shape = (
                self.bond_dim1,
                self.bond_dim1,
                self.phys_d,
                self.phys_d,
                self.bond_dim2,
                self.bond_dim2,
            )
            H_eff = raw_variational_tensor.reshape(shape).transpose((0, 2, 4, 1, 3, 5))
            """
              *-- 0 4 --*
              |    | 2  |
              #----#----#
              |    | 3  |
              *-- 1 5 --*
            """
        else: 
            if Rend: 
                left_middle = np.tensordot(self.left_ms.calc_L_parity(), self.mpo_parity, axes=[3, 0])

#                #test energy each site
#                Etmp = np.tensordot(left_middle, self.matrix, axes=[[4, 7], [0, 1]])
#                E    = np.tensordot(Etmp , self.matrix, axes=[[1, 5], [0, 1]])
#                print(E)

                shape = (
                    self.left_ms.bond_dim2,
                    self.left_ms.bond_dim2,
                    self.phys_d,
                    self.phys_d,
                )
                """
                  *-- 0  
                  |    | 2     
                  #----#
                  |    | 3     
                  *-- 1 
                """
                H_eff = left_middle.reshape(shape).transpose((0, 2, 1, 3))

            else:
                raw_variational_tensor = np.tensordot(
                    self.mpo_parity, self.right_ms.calc_R_parity(), axes=[1, 2]
                )

#                #test energy each site
#                Etmp = np.tensordot(raw_variational_tensor, self.matrix, axes=[[2, 6], [1, 2]])
#                E    = np.tensordot(Etmp , self.matrix, axes=[[1, 2], [1, 2]])
#                print(E)

                shape = (
                    self.phys_d,
                    self.phys_d,
                    self.right_ms.bond_dim1,
                    self.right_ms.bond_dim1,
                )
                """
                   2 --*
                0 |    |
                  #----#
                1 |    |
                   3 --*
                """
                H_eff = raw_variational_tensor.reshape(shape).transpose((0, 2, 1, 3))

        return H_eff

    def variational_update(self, direction, variational_tensor=None):
        """
        Update the matrix of this state to search ground state by variation method
        :param direction: the direction to update. 'right' means from left to right and 'left' means from right to left
        :return the energy of the updated state.
        """
        assert direction == "left" or direction == "right"
        dim = self.bond_dim1 * self.phys_d * self.bond_dim2
#        print(dim)
        # reshape variational tensor to a square matrix
#        variational_tensor = self.calc_variational_tensor().reshape(dim, dim)
        if variational_tensor is None:
            variational_tensor = self.calc_variational_tensor().reshape(dim, dim)
#            variational_tensor = self.calc_variational_tensor_parity().reshape(dim, dim)
        eig_val=0
        # find the smallest eigenvalue and eigenvector. Note the value returned by `eigs` are complex numbers
# test energy each site
#        if 2 < dim:
#            complex_eig_val, complex_eig_vec = sps_eigs(
#                variational_tensor, 1, which="SR"
#            )
#            eig_val = complex_eig_val.real
#            eig_vec = complex_eig_vec.real
#        else:
        all_eig_val, all_eig_vec = np.linalg.eigh(variational_tensor)
        eig_val = all_eig_val[0]
        eig_vec = all_eig_vec[:, 0]

        Lend = self.left_ms.is_sentinel
        Rend = self.right_ms.is_sentinel

        # reshape the eigenvector back to a matrix state
        if not Lend and not Rend:
            self.matrix = eig_vec.reshape(self.bond_dim1, self.phys_d, self.bond_dim2)
            if direction == "right":
                self.left_canonicalize()
            if direction == "left":
                self.right_canonicalize()
# only right2left sweep 
#                self.left_canonicalize()
        else: 
            if Lend: 
                self.matrix = eig_vec.reshape(1, self.phys_d, self.bond_dim2)
                self.left_canonicalize()
            else: 
                self.matrix = eig_vec.reshape(self.bond_dim1, self.phys_d, 1)
                self.right_canonicalize()
# only right2left sweep 
#                self.left_canonicalize()

        # perform normalization
#lsh test: only right 2 left sweep
#        if not (flag_sentinel_l or flag_sentinel_r):

        return float(eig_val)
#        else:
#            if direction == "right":
#                self.right_canonicalize()
#            if direction == "left":
#                self.left_canonicalize()
#            return float(eig_val)

    def insert_ts_before(self, ts):
        """
        insert a matrix state before this matrix state. Standard doubly linked list operation.
        """
        left_ms = self.left_ms
        left_ms.right_ms = ts
        ts.left_ms, ts.right_ms = left_ms, self
        self.left_ms = ts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "MatrixState (%d, %d, %d)" % (
            self.bond_dim1,
            self.phys_d,
            self.bond_dim2,
        )

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """
        :return: True if this state is not a sentinel state and vice versa.
        """
        return not self.is_sentinel


class MatrixProductState(object):
    """
    A doubly linked list of `MatrixState`. The matrix product state of the whole wave function.
    """

    # initial bond dimension when using `error_threshold` as criterion for compression
    initial_bond_dimension = 50

    def __init__(self, WD_list, WL, WR, P, I, max_bond_dimension=None, error_threshold=0):
        """
        Initialize a MatrixProductState with given bond dimension.
        :param WD_list: the list for WD MPOs. The site num depends on the length of the list
        :param max_bond_dimension: the bond dimension required. The higher bond dimension, the higher accuracy and compuational cost
        :param error_threshold: error threshold used in svd compressing of the matrix state.
        The lower the threshold, the higher the accuracy.
        """
        if max_bond_dimension is None and error_threshold == 0:
            raise ValueError(
                "Must provide either `max_bond_dimension` or `error_threshold`. None is provided."
            )
        if max_bond_dimension is not None and error_threshold != 0:
            raise ValueError(
                "Must provide either `max_bond_dimension` or `error_threshold`. Both are provided."
            )
        self.max_bond_dimension = max_bond_dimension
        if max_bond_dimension is not None:
            bond_dim = max_bond_dimension
        else:
            bond_dim = self.initial_bond_dimension
        self.error_threshold = error_threshold
        site_num = len(WD_list)
        self.site_num = site_num 
        self.WL = WL
        self.WR = WR
#        self.WD = WD
        self.P = P 
        self.I = I 
        # establish the sentinels for the doubly linked list
        self.tensor_state_head = MatrixState.create_sentinel()
        self.tensor_state_tail = MatrixState.create_sentinel()
        self.tensor_state_head.right_ms = self.tensor_state_tail
        self.tensor_state_tail.left_ms = self.tensor_state_head
        # generate objects for MPS, MPO at each site 
        # initialize the matrix states with random numbers.
        M_list = (
            [MatrixState(1, bond_dim, WD_list[0], error_threshold, 1, site_num )]
            + [
                MatrixState(bond_dim, bond_dim, WD_list[i+1], error_threshold, i+2, site_num)
                for i in range(site_num - 2)
            ]
            + [MatrixState(bond_dim, 1, WD_list[-1], error_threshold, site_num, site_num)]
        )
        # insert matrix states to the doubly linked list
        for ts in M_list:
            self.tensor_state_tail.insert_ts_before(ts)
        # perform the initial normalization
        self.tensor_state_head.right_ms.left_canonicalize_all()
        # starting from left MPS at the end of right 
        # LLLLLL ... LCR  <-- right 2 left from C
#lsh test: only right 2 left sweep
#        self.tensor_state_tail.left_ms.right_canonicalize()

        
        # test for the unitarity
        # for ts in self.iter_ts_left2right():
        #    ts.test_left_unitary()

    def iter_ms_left2right(self):
        """
        matrix state iterator. From left to right
        """
        ms = self.tensor_state_head.right_ms
        if ms.left_ms.is_sentinel: ms = ms.right_ms
        while ms:
            yield ms
            ms = ms.right_ms

    def iter_ms_right2left(self):
        """
        matrix state iterator. From right to left
        """
        ms = self.tensor_state_tail.left_ms
#lsh test: only right 2 left sweep
#        if ms.right_ms.is_sentinel: ms = ms.left_ms
        while ms:
            yield ms
            ms = ms.left_ms

    def iter_ms_left2right_wo_skip(self):
        """
        matrix state iterator. From left to right
        """
        ms = self.tensor_state_head.right_ms
        while ms:
            yield ms
            ms = ms.right_ms

    def iter_ms_right2left_wo_skip(self):
        """
        matrix state iterator. From right to left
        """
        ms = self.tensor_state_tail.left_ms
        while ms:
            yield ms
            ms = ms.left_ms

    def mpo_WLR_update(self, OBJ, WL, WR, osite, isite, fci=False):
        """
        applying parity on WL[osite]WR[osite+1] mpo at isite th site superblock 
        """
#        print("mpoL0",L[0].mpo_parity.shape) 
#        print("mpoL1",L[1].mpo_parity.shape) 
#        print("mpoL2",L[2].mpo_parity.shape) 
#
#        print("osite=",osite,", isite=", isite)

        # check WL, WR
        argument_error = ValueError(
            "The definition of MPO is incorrect. Datatype: %s, shape: %s."
            "Please make sure it's a numpy array and check the dimensions of the MPO."
            % (type(WL), WL.shape)
        )
        if not isinstance(WL, np.ndarray):
            raise argument_error
        if not isinstance(WR, np.ndarray):
            raise argument_error
        if WL.ndim != 4:
            raise argument_error
        if WR.ndim != 4:
            raise argument_error
        # WL, WR.shape : ( bond_o1, bond_o2, phys_d1, phys_d2 )
        # WL, WR.shape[0] = bond_o1, W.shape[1] = bond_o2, .. )
        if WL.shape[2] != WL.shape[3]:
            raise argument_error
        if WL.shape[0] != WL.shape[1]:
            raise argument_error
        if WR.shape[2] != WR.shape[3]:
            raise argument_error
        if WR.shape[0] != WR.shape[1]:
            raise argument_error

        P = self.P
        I = self.I
#        print("P",P)
#        print("I",I)
        site_num=self.site_num

        if fci is True:

            if osite is 1:
                mpo_1 = WL[0].copy()
                mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
                mpo_2 = np.tensordot(P, WR, 
                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
                mpo_3 = I.copy()
                mpo_4 = I.copy()
                mpo_5 = I[:, -1].copy()
                mpo_5 = mpo_5.reshape((mpo_5.shape[0],) + (1,) + mpo_5.shape[1:])
                # update mpo and applying parity depending on isite
                OBJ[0].mpo_parity = mpo_1
                OBJ[1].mpo_parity = mpo_2 
                OBJ[2].mpo_parity = mpo_3
                OBJ[3].mpo_parity = mpo_4
                OBJ[4].mpo_parity = mpo_5
    
            elif osite is 2:
                # prepare W_start, WL, WR, W_end
                mpo_1 = I[0].copy()
                mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
                mpo_2 = WL.copy()
                mpo_3 = np.tensordot(P, WR, 
                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
                mpo_4 = I.copy()
                mpo_5 = I[:, -1].copy()
                mpo_5 = mpo_5.reshape((mpo_5.shape[0],) + (1,) + mpo_5.shape[1:])
                # update mpo and applying parity depending on isite
                OBJ[0].mpo_parity = mpo_1
                OBJ[1].mpo_parity = mpo_2 
                OBJ[2].mpo_parity = mpo_3
                OBJ[3].mpo_parity = mpo_4
                OBJ[4].mpo_parity = mpo_5
            elif osite is 3:
                mpo_1 = I[0].copy()
                mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
                mpo_2 = I.copy()
                mpo_3 = WL.copy()
                mpo_4 = np.tensordot(P, WR, 
                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
                mpo_5 = I[:, -1].copy()
                mpo_5 = mpo_5.reshape((mpo_5.shape[0],) + (1,) + mpo_5.shape[1:])
                # update mpo and applying parity depending on isite
                OBJ[0].mpo_parity = mpo_1
                OBJ[1].mpo_parity = mpo_2 
                OBJ[2].mpo_parity = mpo_3
                OBJ[3].mpo_parity = mpo_4
                OBJ[4].mpo_parity = mpo_5

            elif osite is 4:
                mpo_1 = I[0].copy()
                mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
                mpo_2 = I.copy()
                mpo_3 = I.copy()
                mpo_4 = WL.copy()
                mpo_5 = WR[:, -1].copy()
                mpo_5 = mpo_5.reshape((mpo_5.shape[0],) + (1,) + mpo_5.shape[1:])
                # update mpo and applying parity depending on isite
                OBJ[0].mpo_parity = mpo_1
                OBJ[1].mpo_parity = mpo_2 
                OBJ[2].mpo_parity = mpo_3
                OBJ[3].mpo_parity = mpo_4
                OBJ[4].mpo_parity = mpo_5

#            if osite is 1:
#                mpo_1 = WL[0].copy()
#                mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
#                mpo_2 = WR.copy()
#                mpo_L = I[:, -1].copy()
#                mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
#                # update mpo and applying parity depending on isite
#                OBJ[0].mpo_parity = mpo_1
#                OBJ[1].mpo_parity = np.tensordot(P, mpo_2, 
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                OBJ[2].mpo_parity = mpo_L
#    
#            elif osite+1 is site_num:
#                # prepare W_start, WL, WR, W_end
#                mpo_S = I[0].copy()
#                mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
#                mpo_1 = WL.copy()
#                mpo_2 = WR[:, -1].copy()
#                mpo_2 = mpo_2.reshape((mpo_2.shape[0],) + (1,) + mpo_2.shape[1:])
#                # update mpo and applying parity depending on isite
#                OBJ[0].mpo_parity = mpo_S 
#                OBJ[1].mpo_parity = mpo_1 
#                OBJ[2].mpo_parity = np.tensordot(P, mpo_2,  
#                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))

            return 

        """
              2
              |
          0 --*-- 1                       1               
              |                           |                           2 
              3                       0 --#-- 2                       |    
                     --tensordot-->       |       --transform-->  0 --#-- 1
              2                           3                           |    
              |                                                       3 
          0 --#-- 1                                              
              |                                                  
              3                                          
        """

#        if osite is 1:
#            # prepare W_start, WL, WR, W_end
#            mpo_1 = WL[0].copy()
#            mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
#            mpo_2 = WR.copy()
#            mpo_L = I[:, -1].copy()
#            mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
#
#            OBJ[0].mpo_parity = mpo_1
#            OBJ[1].mpo_parity = np.tensordot(P, mpo_2, 
#                   axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#            OBJ[2].mpo_parity = mpo_L
#
##            # update mpo and applying parity depending on isite
##            if isite is 1:
##                OBJ[0].mpo_parity = mpo_1
##                OBJ[1].mpo_parity = np.tensordot(P, mpo_2, 
##                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##                OBJ[2].mpo_parity = mpo_L
##            elif isite is 2:
##                OBJ[0].mpo_parity = np.tensordot(mpo_1, P, 
##                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##                OBJ[1].mpo_parity = mpo_2 
##                OBJ[2].mpo_parity = mpo_L
##            else:
##                OBJ[0].mpo_parity = mpo_1
##                OBJ[1].mpo_parity = np.tensordot(P, mpo_2, 
##                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##                OBJ[2].mpo_parity = mpo_L
#
##            print("mpo1",L[0].mpo_parity.shape) 
##            print("mpo2",L[1].mpo_parity.shape) 
##            print("mpo3",L[2].mpo_parity.shape) 
#
#        elif osite+1 is 3:
#            # prepare W_start, WL, WR, W_end
#            mpo_S = I[0].copy()
#            mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
#            mpo_1 = WL.copy()
#            mpo_2 = WR[:, -1].copy()
#            mpo_2 = mpo_2.reshape((mpo_2.shape[0],) + (1,) + mpo_2.shape[1:])
#
##            print("mpoS",mpo_S.shape) 
##            print("mpo1",mpo_1.shape) 
##            print("mpo2",mpo_2.shape) 
#
#            OBJ[0].mpo_parity = mpo_S 
#            OBJ[1].mpo_parity = mpo_1 
#            OBJ[2].mpo_parity = np.tensordot(P, mpo_2,  
#              axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##            # update mpo and applying parity depending on isite
##            if isite is 3:
###                OBJ[0].mpo_parity = mpo_S
##                OBJ[0].mpo_parity = np.tensordot(mpo_S, P,  
##                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##                OBJ[1].mpo_parity = np.tensordot(mpo_1, P,  
##                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##                OBJ[2].mpo_parity = mpo_2
##            elif isite is 2:
###                OBJ[0].mpo_parity = mpo_S
##                OBJ[0].mpo_parity = np.tensordot(mpo_S, P,  
##                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##                OBJ[1].mpo_parity = mpo_1 
##                OBJ[2].mpo_parity = np.tensordot(P, mpo_2,  
##                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##            else:
##                OBJ[0].mpo_parity = mpo_S
##                OBJ[1].mpo_parity = mpo_1 
##                OBJ[2].mpo_parity = np.tensordot(P, mpo_2,  
##                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##            print("mpo1",OBJ[0].mpo_parity.shape) 
##            print("mpo2",OBJ[1].mpo_parity.shape) 
##            print("mpo3",OBJ[2].mpo_parity.shape) 

# parity same as fci
        if osite is 1:
            # prepare W_start, WL, WR, W_end
            mpo_1 = WL[0].copy()
            mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
            mpo_2 = np.tensordot(P, WR, 
                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
            mpo_L = I[:, -1].copy()
            mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
            # update mpo and applying parity depending on isite
            OBJ[0].mpo_parity = mpo_1
            OBJ[1].mpo_parity = mpo_2 
            for itmp in range(site_num-3):
                OBJ[itmp+2].mpo_parity = I.copy() 
            OBJ[site_num-1].mpo_parity = mpo_L

        elif osite > 1 and osite + 1 < site_num:
            # prepare W_start, WL, WR, W_end
            mpo_S = I[0].copy()
            mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
            mpo_1 = WL.copy()
            mpo_2 = np.tensordot(P, WR, 
                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
            mpo_L = I[:, -1].copy()
            mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])

            # update mpo and applying parity depending on isite
            OBJ[0].mpo_parity = mpo_S
            for itmp in range(site_num-2):
                if itmp+2 is osite:
                   OBJ[itmp+1].mpo_parity = mpo_1
                elif itmp+2 is osite+1:
                   OBJ[itmp+1].mpo_parity = mpo_2 
                else: 
                   OBJ[itmp+1].mpo_parity = I.copy() 
            OBJ[site_num-1].mpo_parity = mpo_L

        elif osite+1 is site_num:
            # prepare W_start, WL, WR, W_end
            mpo_1 = WL.copy()
            mpo_2 = WR[:, -1].copy()
            mpo_2 = mpo_2.reshape((mpo_2.shape[0],) + (1,) + mpo_2.shape[1:])
            mpo_2 = np.tensordot(P, mpo_2, 
                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
            if site_num > 2:
                mpo_S = I[0].copy()
                mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
            elif site_num is 2:
                mpo_1 = WL[0].copy()
                mpo_1 = mpo_S.reshape((1,) + mpo_S.shape)

            # update mpo and applying parity depending on isite
            OBJ[0].mpo_parity = mpo_S
            for itmp in range(site_num-3):
                OBJ[itmp+1].mpo_parity = I.copy() 
            OBJ[site_num-2].mpo_parity = mpo_1 
            OBJ[site_num-1].mpo_parity = mpo_2 

# parity depending on i site 
#        if osite is 1:
#            # prepare W_start, WL, WR, W_end
#            mpo_1 = WL[0].copy()
#            mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
#            mpo_2 = WR.copy()
#            if site_num > 2:
#                mpo_L = I[:, -1].copy()
#                mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
#            elif site_num is 2:
#                mpo_L = WR[:, -1].copy()
#                mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
#
#            # update mpo and applying parity depending on isite
#            if isite is osite:
#                OBJ[0].mpo_parity = mpo_1
#                OBJ[1].mpo_parity = np.tensordot(P, mpo_2, 
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                for itmp in range(site_num-3):
#                    OBJ[itmp+2].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#            elif isite is osite+1:
#                OBJ[0].mpo_parity = np.tensordot(mpo_1, P, 
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                OBJ[1].mpo_parity = mpo_2 
#                for itmp in range(site_num-3):
#                    OBJ[itmp+2].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#            else:
#                OBJ[0].mpo_parity = mpo_1
#                OBJ[1].mpo_parity = np.tensordot(P, mpo_2, 
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                for itmp in range(site_num-3):
#                    OBJ[itmp+2].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#
##            print("mpo1",OBJ[0].mpo_parity.shape) 
##            print("mpo2",OBJ[1].mpo_parity.shape) 
##            print("mpo3",OBJ[2].mpo_parity.shape) 
#
#        elif osite > 1 and osite + 1 < site_num:
#            # prepare W_start, WL, WR, W_end
#            mpo_S = I[0].copy()
#            mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
#            mpo_1 = WL.copy()
#            mpo_2 = WR.copy()
#            mpo_L = I[:, -1].copy()
#            mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])
#
##            print("mpoS",mpo_S.shape) 
##            print("mpo1",mpo_1.shape) 
##            print("mpo2",mpo_2.shape) 
##            print("mpoL",mpo_L.shape) 
#
#            # update mpo and applying parity depending on isite
#            if isite < osite:
#                OBJ[0].mpo_parity = mpo_S
#                for itmp in range(site_num-2):
#                    if itmp+1 is osite:
#                        OBJ[itmp+1].mpo_parity = mpo_1
#                    elif itmp+1 is osite+1:
#                        OBJ[itmp+1].mpo_parity = np.tensordot(P, mpo_2, 
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    else: 
#                        OBJ[itmp+1].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#            elif isite is osite:
#                OBJ[0].mpo_parity = np.tensordot(mpo_S, P, 
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                for itmp in range(site_num-2):
#                    if itmp+1 < osite:
#                        OBJ[itmp+1].mpo_parity = np.tensordot(P, I, 
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    elif itmp+1 is osite:
#                        OBJ[itmp+1].mpo_parity = mpo_1
#                    elif itmp+1 is osite+1:
#                        OBJ[itmp+1].mpo_parity = np.tensordot(P, mpo_2, 
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    else: 
#                        OBJ[itmp+1].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#            elif isite is osite+1:
#                OBJ[0].mpo_parity = np.tensordot(mpo_S, P, 
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                for itmp in range(site_num-2):
#                    if itmp+1 < osite:
#                        OBJ[itmp+1].mpo_parity = np.tensordot(P, I, 
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    elif itmp+1 is osite:
#                        OBJ[itmp+1].mpo_parity = np.tensordot(mpo_1, P,  
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    elif itmp+1 is osite+1:
#                        OBJ[itmp+1].mpo_parity = mpo_2
#                    else: 
#                        OBJ[itmp+1].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#            else:
#                OBJ[0].mpo_parity = mpo_S
#                for itmp in range(site_num-2):
#                    if itmp+1 is osite:
#                        OBJ[itmp+1].mpo_parity = mpo_1
#                    elif itmp+1 is osite+1:
#                        OBJ[itmp+1].mpo_parity = np.tensordot(P, mpo_2, 
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    else: 
#                        OBJ[itmp+1].mpo_parity = I.copy() 
#                OBJ[site_num-1].mpo_parity = mpo_L
#
##            print("mpo1",OBJ[0].mpo_parity.shape) 
##            print("mpo2",OBJ[1].mpo_parity.shape) 
##            print("mpo3",OBJ[2].mpo_parity.shape) 
#
#        elif osite+1 is site_num:
#            # prepare W_start, WL, WR, W_end
#            mpo_1 = WL.copy()
#            mpo_2 = WR[:, -1].copy()
#            mpo_2 = mpo_2.reshape((mpo_2.shape[0],) + (1,) + mpo_2.shape[1:])
#            if site_num > 2:
#                mpo_S = I[0].copy()
#                mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
#            elif site_num is 2:
#                mpo_S = WL[0].copy()
#                mpo_S = mpo_S.reshape((1,) + mpo_S.shape)
#
##            print("mpoS",mpo_S.shape) 
##            print("mpo1",mpo_1.shape) 
##            print("mpo2",mpo_2.shape) 
#
#            # update mpo and applying parity depending on isite
#            if isite is osite+1:
#                OBJ[0].mpo_parity = np.tensordot(mpo_S, P,  
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                for itmp in range(site_num-3):
#                    OBJ[itmp+1].mpo_parity = np.tensordot(I, P,  
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                OBJ[site_num-2].mpo_parity = np.tensordot(mpo_1, P,  
#                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                OBJ[site_num-1].mpo_parity = mpo_2
#            elif isite is osite:
#                OBJ[0].mpo_parity = np.tensordot(mpo_S, P,  
#                           axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                for itmp in range(site_num-3):
#                    OBJ[itmp+1].mpo_parity = np.tensordot(I, P,  
#                        axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                OBJ[site_num-2].mpo_parity = mpo_1 
#                OBJ[site_num-1].mpo_parity = np.tensordot(P, mpo_2,  
#                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#            else:
#                OBJ[0].mpo_parity = mpo_S
#                for itmp in range(site_num-3):
#                    OBJ[itmp+2].mpo_parity = I.copy() 
#                OBJ[site_num-2].mpo_parity = mpo_1 
#                OBJ[site_num-1].mpo_parity = np.tensordot(P, mpo_2,  
#                    axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
##            print("mpo1",OBJ[0].mpo_parity.shape) 
##            print("mpo2",OBJ[1].mpo_parity.shape) 
##            print("mpo3",OBJ[2].mpo_parity.shape) 
        return 

    def mpo_parity(self, L, isite):
        """
        applying parity on mpo at i th site superblock 
        """
        P = self.P
        site_num=self.site_num
        """
              2
              |
          0 --*-- 1                       1               
              |                           |                           2 
              3                       0 --#-- 2                       |    
                     --tensordot-->       |       --transform-->  0 --#-- 1
              2                           3                           |    
              |                                                       3 
          0 --#-- 1                                              
              |                                                  
              3                                          
        """
#        if isite is site_num:
#            for itmp in range(site_num):
#                if itmp is 0:
#                    L[itmp].mpo_parity = np.tensordot(L[itmp].mpo, P, 
#                                         axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                elif itmp is site_num-1:
#                    L[itmp].mpo_parity = L[itmp].mpo
#                else:
#                    L[itmp].mpo_parity = np.tensordot(P, L[itmp].mpo, 
#                                         axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#                    L[itmp].mpo_parity = np.tensordot(L[itmp].mpo_parity, P,
#                                         axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
#        else:
#            for itmp in range(site_num):
#                if itmp is 0 or itmp is isite-1:
#                    L[itmp].mpo_parity = L[itmp].mpo
#                else:
#                    L[itmp].mpo_parity = np.tensordot(P, L[itmp].mpo, 
#                                         axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))
        for itmp in range(site_num):
            if itmp is 0:
                L[itmp].mpo_parity = L[itmp].mpo
            else:
                L[itmp].mpo_parity = np.tensordot(P, L[itmp].mpo, 
                                     axes=[[1, 3], [0, 2]]).transpose((0, 2, 1, 3))

        return 

    def search_ground_state(self,num_mpo_list):
        """
        Find the ground state (optimize the energy) of the MPS by variation method
        :return the energies of each step during the optimization
        """
        energies = []
        numelect = []
        OBJ = []
        # stop when the energies does not change anymore
# object list
        OBJ.append(self.tensor_state_head.right_ms)
        for itmp in range(self.site_num-1):
            OBJ.append(OBJ[itmp].right_ms)

        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
            site_num_tmp=len(num_mpo_list)
            it=0
            for itmp in range(self.site_num):
                # right2left
                r2l = self.site_num - itmp - 1
                # update super block operator (SBO) with new mpo with parity
                dim = OBJ[r2l].bond_dim1 * OBJ[r2l].phys_d * OBJ[r2l].bond_dim2
                SBO = OBJ[r2l].calc_variational_tensor().reshape(dim, dim)
                for ltmp in range(site_num_tmp-1):
                    self.mpo_WLR_update(OBJ, self.WL, self.WR, ltmp+1, r2l+1)
                    SBO = SBO + OBJ[r2l].calc_variational_tensor_static(OBJ).reshape(dim, dim)
                energies.append(OBJ[r2l].variational_update("left",SBO))
# energy calculation without update for debug
#                energies.append(self.expectation(self.mpo_list))
                numelect.append(self.expectation(num_mpo_list))
                print(graphic(it,site_num_tmp-it-1,"l"))
                it=it+1
                print(energies[-1], numelect[-1])
            self.tensor_state_head.right_ms.left_canonicalize_all()

#lsh
##        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
#        clock=0
#        while clock is 0: 
#            clock=1
#            site_num_tmp=len(num_mpo_list)
#            it=0
#            for ts in self.iter_ms_right2left():
#                # update super block operator (SBO) with new mpo with parity
##                print(ts.bond_dim1, ts.phys_d, ts.bond_dim2)
##                dim = ts.bond_dim1 * ts.phys_d * ts.bond_dim2
##                SBO = ts.calc_variational_tensor().reshape(dim, dim)
#
##                print(dim, SBO)
##                self.mpo_WLR_update(L, self.WL, self.WR, 1, site_num_tmp-it)
#
##                print(site_num_tmp-it)
##                for ltmp in range(site_num_tmp-1):
##                    self.mpo_WLR_update(L, self.WL, self.WR, ltmp+1, site_num_tmp-it)
##                    SBO = np.add(SBO, ts.calc_variational_tensor().reshape(dim, dim))
#                energies.append(ts.variational_update("left",SBO))
##                energies.append(ts.variational_update("left"))
##                print(self.H_eff_sentinel_test())
## energy calculation without update for debug
##                energies.append(self.expectation(self.mpo_list))
#                numelect.append(self.expectation(num_mpo_list))
##                print(graphic(it,site_num_tmp-it-1,"l"))
#                it=it+1
##                print(ts.is_sentinel, ts.right_ms.is_sentinel, ts.left_ms.is_sentinel)
##                print(energies[-1],numelect[-1])
#                print(energies[-1])
#
#            # perform the initial normalization
#            self.tensor_state_head.right_ms.left_canonicalize_all()
#
##        print(energies[-1],numelect[-1])

##sucess
##lsh test: parity dmrg test for 3 site model
#        L1=self.tensor_state_head.right_ms
#        L2=L1.right_ms
#        L3=L2.right_ms
#        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
##            site_num_tmp=len(num_mpo_list)
#            it=0
#            for ts in self.iter_ms_right2left():
#                L1.mpo_parity, L2.mpo_parity, L3.mpo_parity = \
#                  self.parity_test_2(L1.mpo,L2.mpo,L3.mpo,it)
##                  self.parity_test_2(L1.mpo,L2.mpo,L3.mpo,self.site_num-it-1)
#                energies.append(ts.variational_update("left"))
#                numelect.append(self.expectation(num_mpo_list))
##                print(graphic(it,site_num_tmp-it-1,"l"))
#                it=it+1
#                print(energies[-1])
#            # perform the initial normalization
#            self.tensor_state_head.right_ms.left_canonicalize_all()
##        print(energies[-1],numelect[-1])

##lsh test: energy calculation on each site 
#        print(self.expectation(self.mpo_list))
#        L3 = self.tensor_state_tail.left_ms
#        L2 = L3.left_ms
#        L1 = L2.left_ms
##        print(self.H_eff_sentinel_test_R())
#        energies.append(L3.variational_update("left"))
#        energies.append(L2.variational_update("left"))
#        energies.append(L1.variational_update("left"))
#
#        L3 = self.tensor_state_head.right_ms
#        L2 = L3.right_ms
#        L1 = L2.right_ms
##        print(self.H_eff_sentinel_test_R())
#        energies.append(L3.variational_update("right"))
#        energies.append(L2.variational_update("right"))
#        energies.append(L1.variational_update("right"))

#lsh test: debug sentinel
#        self.tensor_state_tail.left_ms.right_canonicalize_all()
#        L1 = self.tensor_state_head.right_ms
#        print(self.H_eff_sentinel_test_L())
#        energies.append(L1.variational_update("right"))
#        print(energies[-1])

##lsh test: oscillating sweep
#        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
#            site_num_tmp=len(num_mpo_list)
#            it=0
#            for ts in self.iter_ms_right2left():
#                energies.append(ts.variational_update("left"))
#                numelect.append(self.expectation(num_mpo_list))
#                it=it+1
##                print(graphic(it,site_num_tmp-it-1,"l"))
##                print(ts.is_sentinel, ts.right_ms.is_sentinel, ts.left_ms.is_sentinel)
##                print(energies[-1],numelect[-1])
#                print(energies[-1])
#            it=0
#            for ts in self.iter_ms_left2right():
#                energies.append(ts.variational_update("right"))
#                numelect.append(self.expectation(num_mpo_list))
#                it=it+1
##                print(graphic(it,site_num_tmp-it-1,"r"))
##                print(ts.is_sentinel, ts.right_ms.is_sentinel, ts.left_ms.is_sentinel)
##                print(energies[-1],numelect[-1])
#                print(energies[-1])

        return energies, numelect

#    def search_ground_state(self):
#        """
#        Find the ground state (optimize the energy) of the MPS by variation method
#        :return the energies of each step during the optimization
#        """
#        energies = []
#        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
#            for ts in self.iter_ms_right2left():
#                energies.append(ts.variational_update("left"))
#            for ts in self.iter_ms_left2right():
#                energies.append(ts.variational_update("right"))
#        return energies

    def full_ci_test(self):
        """
        Find the ground state (optimize the energy) by direct diagonalization of H 
        """
#        energies = []

        L3 = self.tensor_state_tail.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo3 = L3.mpo
        mpo2 = L2.mpo
        mpo1 = L1.mpo

#        L1.mpo_parity, L2.mpo_parity, L3.mpo_parity = self.parity_test(mpo1, mpo2, mpo3)
#
#        mpo3 = L3.mpo_parity
#        mpo2 = L2.mpo_parity
#        mpo1 = L1.mpo_parity

        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12  = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123 = np.tensordot(H12,  mpo3, axes=[[3], [0]])

        phys_d = L2.phys_d
        """
           0     2     4     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           1     3     5     
        """
        shape = (
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
        )
        """
           0     1     2     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           3     4     5     
        """
        H123t = H123.reshape(shape).transpose((0, 2, 4, 1, 3, 5))
        dim = phys_d * phys_d * phys_d 
        H123f = H123t.reshape(dim, dim)

        all_eig_val, all_eig_vec = np.linalg.eigh(H123f)
        eig_val = all_eig_val[0]
        eig_vec = all_eig_vec[:, 0]

        return float(eig_val) 

#        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
#            for ts in self.iter_ms_right2left():
#                energies.append(ts.variational_update("left"))
#            for ts in self.iter_ms_left2right():
#                energies.append(ts.variational_update("right"))
#        return energies

    def full_ci_mat(self):
        """
        Find the ground state (optimize the energy) by direct diagonalization of H 
        """
#        energies = []

        L3 = self.tensor_state_tail.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo3 = L3.mpo
        mpo2 = L2.mpo
        mpo1 = L1.mpo
#        print(mpo1)
#        print(mpo2)
#        print(mpo3)

#        L1.mpo_parity, L2.mpo_parity, L3.mpo_parity = self.parity_test(mpo1, mpo2, mpo3)
#
#        mpo3 = L3.mpo_parity
#        mpo2 = L2.mpo_parity
#        mpo1 = L1.mpo_parity

        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12  = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123 = np.tensordot(H12,  mpo3, axes=[[3], [0]])

        phys_d = L2.phys_d
        """
           0     2     4     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           1     3     5     
        """
        shape = (
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
        )
        """
           0     1     2     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           3     4     5     
        """
        H123t = H123.reshape(shape).transpose((0, 2, 4, 1, 3, 5))

        return H123t

    def full_ci_mat_5(self):
        """
        Find the ground state (optimize the energy) by direct diagonalization of H 
        """
#        energies = []

        L5 = self.tensor_state_tail.left_ms
        L4 = L5.left_ms
        L3 = L4.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo5 = L5.mpo
        mpo4 = L4.mpo
        mpo3 = L3.mpo
        mpo2 = L2.mpo
        mpo1 = L1.mpo
#        print(mpo1)
#        print(mpo2)
#        print(mpo3)

#        L1.mpo_parity, L2.mpo_parity, L3.mpo_parity = self.parity_test(mpo1, mpo2, mpo3)
#
#        mpo3 = L3.mpo_parity
#        mpo2 = L2.mpo_parity
#        mpo1 = L1.mpo_parity

        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12   = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123  = np.tensordot(H12,  mpo3, axes=[[3], [0]])
        H1234 = np.tensordot(H123, mpo4, axes=[[5], [0]])
        H12345= np.tensordot(H1234,mpo5, axes=[[7], [0]])

        phys_d = L2.phys_d
        shape = (
            phys_d, phys_d,
            phys_d, phys_d,
            phys_d, phys_d,
            phys_d, phys_d,
            phys_d, phys_d,
        )
        H12345t = H12345.reshape(shape).transpose((0, 2, 4, 6, 8, 1, 3, 5, 7, 9))

        return H12345t

    def full_ci_mat_5_parity(self):
        """
        Find the ground state (optimize the energy) by direct diagonalization of H 
        """
#        energies = []

        L5 = self.tensor_state_tail.left_ms
        L4 = L5.left_ms
        L3 = L4.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo5 = L5.mpo_parity
        mpo4 = L4.mpo_parity
        mpo3 = L3.mpo_parity
        mpo2 = L2.mpo_parity
        mpo1 = L1.mpo_parity


        print(mpo1.shape)
        print(mpo2.shape)
        print(mpo3.shape)
        print(mpo4.shape)
        print(mpo5.shape)

#        print(mpo1)
#        print(mpo2)
#        print(mpo3)

#        L1.mpo_parity, L2.mpo_parity, L3.mpo_parity = self.parity_test(mpo1, mpo2, mpo3)
#
#        mpo3 = L3.mpo_parity
#        mpo2 = L2.mpo_parity
#        mpo1 = L1.mpo_parity

        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12   = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123  = np.tensordot(H12,  mpo3, axes=[[3], [0]])
        H1234 = np.tensordot(H123, mpo4, axes=[[5], [0]])
        H12345= np.tensordot(H1234,mpo5, axes=[[7], [0]])

        phys_d = L2.phys_d
        shape = (
            phys_d, phys_d,
            phys_d, phys_d,
            phys_d, phys_d,
            phys_d, phys_d,
            phys_d, phys_d,
        )
        H12345t = H12345.reshape(shape).transpose((0, 2, 4, 6, 8, 1, 3, 5, 7, 9))

        return H12345t

    def full_ci_mat_parity(self):
        """
        Find the ground state (optimize the energy) by direct diagonalization of H 
        """
#        energies = []

        L3 = self.tensor_state_tail.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo3 = L3.mpo_parity
        mpo2 = L2.mpo_parity
        mpo1 = L1.mpo_parity

#        L1.mpo_parity, L2.mpo_parity, L3.mpo_parity = self.parity_test(mpo1, mpo2, mpo3)
#
#        mpo3 = L3.mpo_parity
#        mpo2 = L2.mpo_parity
#        mpo1 = L1.mpo_parity

        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12  = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123 = np.tensordot(H12,  mpo3, axes=[[3], [0]])

        phys_d = L2.phys_d
        """
           0     2     4     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           1     3     5     
        """
        shape = (
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
        )
        """
           0     1     2     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           3     4     5     
        """
        H123t = H123.reshape(shape).transpose((0, 2, 4, 1, 3, 5))

        return H123t

    def parity_test(self, mpo1, mpo2, mpo3):
        """
        
        
        """
        P = self.P
        """

              2
              |
          0 --*-- 1                       1               
              |                           |                           2 
              3                       0 --#-- 2                       |    
                     --tensordot-->       |       --tensordot-->  0 --#-- 1
              2                           3                           |    
              |                                                       3 
          0 --#-- 1                                              
              |                                                  
              3                                          
        """
        # up_middle is of shape (1, 5, 2, 3, 4)
#        up_middle = np.tensordot(self.matrix.conj(), mpo, axes=[1, 2])

        mpo1_parity = mpo1
#        mpo2_parity = mpo2
#        mpo3_parity = mpo3
#        mpo1_parity = np.tensordot(mpo1, P, axes=[[1, 3], [0, 2]]).transpose(
#                      (0, 2, 1, 3))
        mpo2_parity = np.tensordot(P, mpo2, axes=[[1, 3], [0, 2]]).transpose(
                      (0, 2, 1, 3))
        mpo3_parity = np.tensordot(P, mpo3, axes=[[1, 3], [0, 2]]).transpose(
                      (0, 2, 1, 3))
        return mpo1_parity, mpo2_parity, mpo3_parity 

    def parity_test_2(self, mpo1, mpo2, mpo3, ityp):
        """
        parity dmrg test for 3 site model
        """
        P = self.P
        """

              2
              |
          0 --*-- 1                       1               
              |                           |                           2 
              3                       0 --#-- 2                       |    
                     --tensordot-->       |       --tensordot-->  0 --#-- 1
              2                           3                           |    
              |                                                       3 
          0 --#-- 1                                              
              |                                                  
              3                                          
        """

#        if ityp is 0:
#            mpo1_parity = mpo1
#            mpo2_parity = np.tensordot(P, mpo2, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))
#            mpo3_parity = np.tensordot(P, mpo3, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))
#        elif ityp is 1:
#            mpo1_parity = np.tensordot(mpo1, P, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))
#            mpo2_parity = mpo2
#            mpo3_parity = np.tensordot(P, mpo3, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))
#        else:
#            mpo3_parity = mpo3
#            mpo1_parity = np.tensordot(mpo1, P, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))
#            mpo2_parity = np.tensordot(P, mpo2, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))
#            mpo2_parity = np.tensordot(mpo2_parity, P, axes=[[1, 3], [0, 2]]).transpose(
#                          (0, 2, 1, 3))

        mpo1_parity = mpo1 
        mpo2_parity = np.tensordot(P, mpo2, axes=[[1, 3], [0, 2]]).transpose(
                      (0, 2, 1, 3))
        mpo3_parity = np.tensordot(P, mpo3, axes=[[1, 3], [0, 2]]).transpose(
                      (0, 2, 1, 3))
        return mpo1_parity, mpo2_parity, mpo3_parity 

    def H_eff_sentinel_test_R(self):
        """
        confirm
        """
#        energies = []

        L3 = self.tensor_state_tail.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo3 = L3.mpo
        mpo2 = L2.mpo
        mpo1 = L1.mpo

        mps3 = L3.matrix
        mps2 = L2.matrix
        mps1 = L1.matrix
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12  = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123 = np.tensordot(H12,  mpo3, axes=[[3], [0]])

        phys_d    = L2.phys_d
        bond_dim2 = L2.bond_dim2
        bond_dim1 = L1.bond_dim1
        """
           0     2     4     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           1     3     5     
        """
        shape = (
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
        )
        """
           0     1     2     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           3     4     5     
        """
        H123t = H123.reshape(shape).transpose((0, 2, 4, 1, 3, 5))

        contr1= np.tensordot(H123t , mps2, axes=[[4], [1]])
        contr2= np.tensordot(contr1, mps1, axes=[[3,5], [1,2]])
        contr3= np.tensordot(contr2, mps2, axes=[[1], [1]])
        contr4= np.tensordot(contr3, mps1, axes=[[0,5], [1,2]])

        shape = (
            phys_d,
            phys_d,
            bond_dim2,
            bond_dim2,
        )      
        H_eff= contr4.reshape(shape).transpose((3, 0, 2, 1))

        dim = phys_d * bond_dim2 
        H_efff= H_eff.reshape(dim, dim)

        all_eig_val, all_eig_vec = np.linalg.eigh(H_efff)
        eig_val = all_eig_val[0]
        eig_vec = all_eig_vec[:, 0]
#        print(phys_d,bond_dim2)
#        print(eig_vec)

        return float(eig_val) 

    def H_eff_sentinel_test_L(self):
        """
        confirm
        """
#        energies = []

        L3 = self.tensor_state_tail.left_ms
        L2 = L3.left_ms
        L1 = L2.left_ms

        mpo3 = L3.mpo
        mpo2 = L2.mpo
        mpo1 = L1.mpo

        mps3 = L3.matrix
        mps2 = L2.matrix
        mps1 = L1.matrix
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              2                  2                           1     4                         1     3     6     
              |                  |                           |     |                         |     |     |     
          0 --#-- 1     +    0 --#-- 1  --tensordot-->   0 --#--#--#-- 3  --reshape-->   0 --#--#--#--#--#-- 5 
              |                  |                           |     |                         |     |     |     
              3                  3                           2     5                         2     4     7     
       
        """
        H12  = np.tensordot(mpo1, mpo2, axes=[[1], [0]])
        H123 = np.tensordot(H12,  mpo3, axes=[[3], [0]])

        phys_d    = L1.phys_d
        bond_dim1 = L2.bond_dim1
        bond_dim2 = L3.bond_dim2
        """
           0     2     4     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           1     3     5     
        """
        shape = (
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
            phys_d,
        )
        """
           0     1     2     
           |     |     |     
         --#--#--#--#--#--  
           |     |     |     
           3     4     5     
        """
        H123t = H123.reshape(shape).transpose((0, 2, 4, 1, 3, 5))

        contr1= np.tensordot(H123t , mps2, axes=[[4], [1]])
        contr2= np.tensordot(contr1, mps3, axes=[[6,4], [0,1]])
        contr3= np.tensordot(contr2, mps2, axes=[[1], [1]])
        contr4= np.tensordot(contr3, mps3, axes=[[6,1], [0,1]])

        shape = (
            phys_d,
            phys_d,
            bond_dim1,
            bond_dim1,
        )      
        H_eff= contr4.reshape(shape).transpose((0, 3, 1, 2))

        dim = phys_d * bond_dim1 
        H_efff= H_eff.reshape(dim, dim)

        all_eig_val, all_eig_vec = np.linalg.eigh(H_efff)
        eig_val = all_eig_val[0]
        eig_vec = all_eig_vec[:, 0]
#        print(phys_d,bond_dim1)
#        print(eig_vec)

        return float(eig_val) 
    def expectation(self, mpo_list):
        """
        Calculate the expectation value of the matrix product state for a certain operator defined in `mpo_list`
        :param mpo_list: a list of mpo from left to right. Construct the MPO by `build_mpo_list` is recommended.
        :return: the expectation value
        """
        F_list = [
            ms.calc_F(mpo) for mpo, ms in zip(mpo_list, self.iter_ms_left2right_wo_skip())
        ]

        def contractor(tensor1, tensor2):
            return np.tensordot(
                tensor1, tensor2, axes=[[1, 3, 5], [0, 2, 4]]
            ).transpose((0, 3, 1, 4, 2, 5))

        expectation = reduce(contractor, F_list).reshape(1)[0]
        return expectation

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "MatrixProductState: %s" % (
            "-".join([str(ms.bond_dim2) for ms in self.iter_ms_left2right_wo_skip()][:-1])
        )

def build_W_list(single_mpo, site_num, regularize=False):
    """
    build MPO list for MPS.
    :param single_mpo: a numpy ndarray with ndim=4.
    The first 2 dimensions reprsents the square shape of the MPO and the last 2 dimensions are physical dimensions.
    :param site_num: the total number of sites
    :param regularize: whether regularize the mpo so that it represents the average over all sites.
    :return MPO list
    """
    argument_error = ValueError(
        "The definition of MPO is incorrect. Datatype: %s, shape: %s."
        "Please make sure it's a numpy array and check the dimensions of the MPO."
        % (type(single_mpo), single_mpo.shape)
    )
    if not isinstance(single_mpo, np.ndarray):
        raise argument_error
    if single_mpo.ndim != 4:
        raise argument_error
    # single_mpo.shape : ( bond_o1, bond_o2, phys_d1, phys_d2 )
    # single_mpo.shape[0] = bond_o1, single_mpo.shape[1] = bond_o2, .. )
    if single_mpo.shape[0] != single_mpo.shape[1]:
        raise argument_error
    # the first MPO, only contains the last row
    mpo_1 = single_mpo[-1].copy()
    mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
    # the last MPO, only contains the first column
    mpo_L = single_mpo[:, 0].copy()
    if regularize:
        mpo_L /= site_num
    mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])

    return [mpo_1] + [single_mpo.copy() for i in range(site_num - 2)] + [mpo_L]

def build_mpo_list(single_mpo, site_num, regularize=False):
    """
    build MPO list for MPS.
    :param single_mpo: a numpy ndarray with ndim=4.
    The first 2 dimensions reprsents the square shape of the MPO and the last 2 dimensions are physical dimensions.
    :param site_num: the total number of sites
    :param regularize: whether regularize the mpo so that it represents the average over all sites.
    :return MPO list
    """
    argument_error = ValueError(
        "The definition of MPO is incorrect. Datatype: %s, shape: %s."
        "Please make sure it's a numpy array and check the dimensions of the MPO."
        % (type(single_mpo), single_mpo.shape)
    )
    if not isinstance(single_mpo, np.ndarray):
        raise argument_error
    if single_mpo.ndim != 4:
        raise argument_error
    # single_mpo.shape : ( bond_o1, bond_o2, phys_d1, phys_d2 )
    # single_mpo.shape[0] = bond_o1, single_mpo.shape[1] = bond_o2, .. )
    if single_mpo.shape[2] != single_mpo.shape[3]:
        raise argument_error
    if single_mpo.shape[0] != single_mpo.shape[1]:
        raise argument_error
    # the first MPO, only contains the last row
    mpo_1 = single_mpo[0].copy()
    mpo_1 = mpo_1.reshape((1,) + mpo_1.shape)
    # the last MPO, only contains the first column
    mpo_L = single_mpo[:,-1].copy()
    if regularize:
        mpo_L /= site_num
    mpo_L = mpo_L.reshape((mpo_L.shape[0],) + (1,) + mpo_L.shape[1:])

    return [mpo_1] + [single_mpo.copy() for i in range(site_num - 2)] + [mpo_L]
