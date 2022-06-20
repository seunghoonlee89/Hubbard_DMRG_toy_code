# 1D Hubbard model / DMRG based on MPS MPO

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mps import MatrixProductState, build_mpo_list

from mpo import *


def construct_W(t=1.0, u=2.0):
    """
    Construct single site mpo based on Hubbard model
    :param t: hopping integral 
    :param u: on-site repulsion 
    :return: constructed single site mpo with shape (1, 6, 4, 4)
    """
    # WL shape: 6, 6, 4, 4
    mpo_block1 = [OI, Opu, Opd, Ou, Od, O0]
    mpo_block2 = [O0,O0,O0,O0,O0,O0]
    mpo_block3 = [O0,O0,O0,O0,O0,O0]
    mpo_block4 = [O0,O0,O0,O0,O0,O0]
    mpo_block5 = [O0,O0,O0,O0,O0,O0]
    mpo_block6 = [O0,O0,O0,O0,O0,O0]
    WL = np.float64([mpo_block1, mpo_block2, mpo_block3, mpo_block4, mpo_block5, mpo_block6])
    WL.flags.writeable = False

    # WR shape: 6, 6, 4, 4
    mpo_block1 = [O0,O0,O0,O0,O0,     O0]
    mpo_block2 = [O0,O0,O0,O0,O0,-t * Ou]
    mpo_block3 = [O0,O0,O0,O0,O0,-t * Od]
    mpo_block4 = [O0,O0,O0,O0,O0, t * Opu]
    mpo_block5 = [O0,O0,O0,O0,O0, t * Opd]
    mpo_block6 = [O0,O0,O0,O0,O0,     OI]
    WR = np.float64([mpo_block1, mpo_block2, mpo_block3, mpo_block4, mpo_block5, mpo_block6])
    WR.flags.writeable = False

    # P shape: 6, 6, 4, 4
    mpo_block1 = [OI,O0,O0,O0,O0,O0]
    mpo_block2 = [O0,OIp,O0,O0,O0,O0]
    mpo_block3 = [O0,O0,OIp,O0,O0,O0]
    mpo_block4 = [O0,O0,O0,OIp,O0,O0]
    mpo_block5 = [O0,O0,O0,O0,OIp,O0]
    mpo_block6 = [O0,O0,O0,O0,O0,OI]
    P = np.float64([mpo_block1, mpo_block2, mpo_block3, mpo_block4, mpo_block5, mpo_block6])
    P.flags.writeable = False

    # I shape: 6, 6, 4, 4
    mpo_block1 = [OI,O0,O0,O0,O0,O0]
    mpo_block2 = [O0,OI,O0,O0,O0,O0]
    mpo_block3 = [O0,O0,OI,O0,O0,O0]
    mpo_block4 = [O0,O0,O0,OI,O0,O0]
    mpo_block5 = [O0,O0,O0,O0,OI,O0]
    mpo_block6 = [O0,O0,O0,O0,O0,OI]
    I = np.float64([mpo_block1, mpo_block2, mpo_block3, mpo_block4, mpo_block5, mpo_block6])
    I.flags.writeable = False

    # WD shape: 2, 2, 4, 4
    mpo_block1 = [OI, u * On]
    mpo_block2 = [O0, OI]

    WD = np.float64([mpo_block1, mpo_block2])
    WD.flags.writeable = False

    return WL,WR,WD,P,I 

def construct_num_mpo():
    """
    Construct single site mpo for number operator
    :return: constructed single site mpo with shape (2, 2, 4, 4)
    """
#    # MPO line by line
    mpo_block1 = [OI, Onum]
    mpo_block2 = [O0, OI]

    # MPO shape: 6, 6, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2])
    # prohibit writing just in case
    mpo.flags.writeable = False

    return mpo

def construct_num_mpo_a():
    """
    Construct single site mpo for number operator
    :return: constructed single site mpo with shape (2, 2, 4, 4)
    """
#    # MPO line by line
    mpo_block1 = [OI, Onuma]
    mpo_block2 = [O0, OI]

    # MPO shape: 6, 6, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2])
    # prohibit writing just in case
    mpo.flags.writeable = False

    return mpo

def construct_num_mpo_b():
    """
    Construct single site mpo for number operator
    :return: constructed single site mpo with shape (2, 2, 4, 4)
    """
#    # MPO line by line
    mpo_block1 = [OI, Onumb]
    mpo_block2 = [O0, OI]

    # MPO shape: 6, 6, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2])
    # prohibit writing just in case
    mpo.flags.writeable = False

    return mpo

def construct_num_mpo22():
    """
    Construct single site mpo for number operator
    :return: constructed single site mpo with shape (2, 2, 4, 4)
    """
    # MPO line by line
    mpo_block1 = [OI22, Onum22]
    mpo_block2 = [O022, OI22]

    # MPO shape: 6, 6, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2])
    # prohibit writing just in case
    mpo.flags.writeable = False

    return mpo

SITE_NUM = 5 
BOND_DIMENSION = 40
print("Hubbard model")
print("SITE NUM =", SITE_NUM)
print("BOND DIM =", BOND_DIMENSION)

# the threshold for error when compressing MPS
ERROR_THRESHOLD = 1e-5

if __name__ == "__main__":
#   CONSTRUCT MPO
    print("build MPO")
    WL, WR, WD, P, I = construct_W()   
    WD_list = build_mpo_list(WD, SITE_NUM)                            # HB OP MPO
    num_mpo_list   = build_mpo_list(construct_num_mpo(), SITE_NUM)    # NUM OP MPO
    num_mpo_list_a = build_mpo_list(construct_num_mpo_a(), SITE_NUM)  # NUM_A OP MPO
    num_mpo_list_b = build_mpo_list(construct_num_mpo_b(), SITE_NUM)  # NUM_B OP MPO

#   CONSTRUCT MPS 
    print("build MPS")
    mps = MatrixProductState(WD_list, WL, WR, P, I, max_bond_dimension=BOND_DIMENSION)
#    mps = MatrixProductState(WD_list, WL, WR, P, I, error_threshold=ERROR_THRESHOLD)

    print("DMRG for Hubbard model (no U1/SU2 symmetry)")
    mps.search_ground_state(num_mpo_list)
    print('')
    print('elec num: tot, alpha, beta')
    print(mps.expectation(num_mpo_list), mps.expectation(num_mpo_list_a), mps.expectation(num_mpo_list_b))
