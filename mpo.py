import numpy as np

# Hubbard MPO matrices
# 0: |0>, 1: |up>, 2: |down>, 3: |up down>

# zero matrix block
O0 = np.zeros((4, 4))
# identity matrix block
OI = np.eye((4))
# a^+_up
Opu = np.float64([[0, 0, 0, 0], [1, 0, 0, 0],[0, 0, 0, 0],[0, 0, 1, 0]])
# a^+_down
Opd = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[1, 0, 0, 0],[0,-1, 0, 0]])
# a_up
Ou = np.float64([[0, 1, 0, 0], [0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 0, 0]])
# a_down
Od = np.float64([[0, 0, 1, 0], [0, 0, 0,-1],[0, 0, 0, 0],[0, 0, 0, 0]])
# n_up n_down 
On = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]])

# number operator
# n_up + n_down
Onum  = np.float64([[0, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 2]])
Onumsq= np.float64([[0, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 4]])
Onuma = np.float64([[0, 0, 0, 0], [0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]])
Onumb = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

#Onum  = np.float64([[0, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 2]])
#Onuma = np.float64([[0, 0, 0, 0], [0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])
#Onumb = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 0]])
#Onumc = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1]])

# parity matrix
Oparity = np.float64([[1, 0, 0, 0], [0,-1, 0, 0],[0, 0,-1, 0],[0, 0, 0, 1]])

# zero matrix block
O022 = np.zeros((2, 2))
# identity matrix block
OI22 = np.eye((2))
Onum22 = np.float64([[0, 0], [0, 1]])

# Parity matrix 
OIp = np.float64([[1, 0, 0, 0], [0,-1, 0, 0],[0, 0,-1, 0],[0, 0, 0, 1]])
# a^+_up
Opupl = np.float64([[0, 0, 0, 0], [-1, 0, 0, 0],[0, 0, 0, 0],[0, 0, 1, 0]])
# a^+_down
Opdpl = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[-1, 0, 0, 0],[0,-1, 0, 0]])
# a_up
Oupl = np.float64([[0, 1, 0, 0], [0, 0, 0, 0],[0, 0, 0, -1],[0, 0, 0, 0]])
# a_down
Odpl = np.float64([[0, 0, 1, 0], [0, 0, 0,1],[0, 0, 0, 0],[0, 0, 0, 0]])

# a^+_up
Opupr = np.float64([[0, 0, 0, 0], [1, 0, 0, 0],[0, 0, 0, 0],[0, 0, -1, 0]])
# a^+_down
Opdpr = np.float64([[0, 0, 0, 0], [0, 0, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0]])
# a_up
Oupr = np.float64([[0, -1, 0, 0], [0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 0, 0]])
# a_down
Odpr = np.float64([[0, 0,-1, 0], [0, 0, 0,-1],[0, 0, 0, 0],[0, 0, 0, 0]])
