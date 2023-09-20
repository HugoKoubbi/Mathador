import json
import os.path as osp
import random
import numpy as np
from typing import Union
from numpy.linalg import det,norm,inv 


#################### Matrix generation ####################
def generate_matrix(n):
    """Generate a column of size n with k non-zero entries"""
    col = np.random.rand(n,n)
    return col

def generate_column_int(k,n):
    L=[]
    for i in range(n):
        L.append(random.randint(0,k))
    return np.array(L)

def generate_matrix_int(k,n):
    """Generate a matrix of size n with k non-zero entries"""
    M = np.zeros((n,n))
    for i in range(n):
        M[i] = generate_column_int(k,n)
    return M    

############ Creation of the dataset via Matrix operations  ####################

###Addition of a matrix 
data_add=[]
pairs = \
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)] 
for m1,m2 in pairs:

    answer = m1 + m2
        
    question = f"{m1} + {m2}" 
    output = f"{m1} + {m2} = {answer}"
    
    assert(output.split()[-1] == str(answer))
    data_add.append({"input": question, "output": output, "answer": str(answer)})

with open("dataset.json", "w") as f:
    json.dump(data_add, f, indent=4)

####Subtraction of a matrix
data_sub=[]
pairs = \
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)] 
for m1,m2 in pairs:

    answer = m1 - m2
        
    question = f"{m1} - {m2}" 
    output = f"{m1} - {m2} = {answer}"
    
    assert(output.split()[-1] == str(answer))
    data_sub.append({"input": question, "output": output, "answer": str(answer)})

with open("dataset.json", "w") as f:
    json.dump(data_sub, f, indent=4)

###Multiplication of a matrix by a matrix
data_mult=[]
pairs = \
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n), generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)] 
for m1,m2 in pairs:

    answer = np.dot(m1,m2)
        
    question = f"{m1} * {m2}" 
    output = f"{m1} * {m2} = {answer}"
    
    assert(output.split()[-1] == str(answer))
    data_mult.append({"input": question, "output": output, "answer": str(answer)})

with open("dataset.json", "w") as f:
    json.dump(data_mult, f, indent=4)

### Trace of a matrix
data_tr=[]
pairs = \
[(generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)] 
for m1 in pairs:

    answer = np.trace(m1)
        
    question = f"trace({m1})" 
    output = f"trace({m1}) = {answer}"
    
    assert(output.split()[-1] == str(answer))
    data_add.append({"input": question, "output": output, "answer": str(answer)})



### Norm of a matrix

data_norm=[]
pairs = \
[(generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)] 
for m1 in pairs:

    answer = np.linalg.norm(m1)
        
    question = f"norm({m1})" 
    output = f"norm({m1}) = {answer}"
    
    assert(output.split()[-1] == str(answer))
    data_norm.append({"input": question, "output": output, "answer": str(answer)})

with open("dataset.json", "w") as f:
    json.dump(data_norm, f, indent=4)


###Determinant of a matrix
data_det=[]
pairs = \
[(generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)] 
for m1 in pairs:

    answer = np.linalg.det(m1)
        
    question = f"det({m1})" 
    output = f"det({m1}) = {answer}"
    
    assert(output.split()[-1] == str(answer))
    data_det.append({"input": question, "output": output, "answer": str(answer)})

with open("dataset.json", "w") as f:
    json.dump(data_det, f, indent=4)

###Inverse of a matrix
data_inv=[]
pairs = \
[(generate_matrix_int(i,n)) for i in range(1,10)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(10,50)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(50,100)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(100,1000)  for n in range(100)] +\
[(generate_matrix_int(i,n)) for i in range(1000,10000)  for n in range(100)]
for m1 in pairs:
    if np.linalg.det(m1)!=0:
        answer = np.linalg.inv(m1)
    else:
        answer = "None"
    question = f"inv({m1})"
    output = f"inv({m1}) = {answer}"

    assert(output.split()[-1] == str(answer))
    data_inv.append({"input": question, "output": output, "answer": str(answer)})
with open("dataset.json", "w") as f:    
    json.dump(data_inv, f, indent=4)
