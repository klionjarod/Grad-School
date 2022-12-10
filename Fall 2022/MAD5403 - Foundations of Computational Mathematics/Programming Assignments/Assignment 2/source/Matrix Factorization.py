#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import os

data = '..\\data\\'


# ## Step 1: Generate a matrix *A* from L and U

# In[4]:


def genRandLUMats(n: int, unit = False):
    """
    Generates a random square matrix A of size n from a unit lower triangular and upper triangular matrix
    
    Params:
        n: integer of the size of the square matrix
        
    Return: 
        nonsingular unit lower triangular matrix, L, upper triangular matrix, U, and their product A
    """
    #L = np.round(np.random.rand(n,n),3) #create an array of random floats with shape (n,n)
    L, U = np.empty((n, n)), np.empty((n, n)) #create empty arrays of shape (n, n)
    
    for row in range(n):
        for col in range(n):
            if row > col:
                U[row][col] = 0
                L[row][col] = np.random.rand(1)
            else:
                U[row][col] = np.random.randint(1, 2)
                L[row][col] = 0
    if unit:
        for i in range(n):
            L[i][i] = 1 #fill diagonals with 1 for unit matrix
    A = L @ U
    return A, L, U 


# In[5]:


A_10, lwr_10, upp_10 = genRandLUMats(10, unit = True)
#ensure A is diagonally dominant
for i in range(10):
    A_10[i][i] += 3
A_100, lwr_100, upp_100 = genRandLUMats(100, unit = True)
#ensure A is diagonally dominant
for i in range(100):
    A_100[i][i] += 3

#check condition number of generated matrices
print(np.linalg.norm(A_10, 2)*np.linalg.norm(np.linalg.inv(A_10), 2))
print(np.linalg.norm(A_100, 2)*np.linalg.norm(np.linalg.inv(A_100), 2))


# In[6]:


def genSymPosMatrix(n: int):
    """
    Generates a symmetric positive square matrix A of size n from a lower triangular matrix and its transpose
    
    Params:
        n: desired rows/cols of the matrix
        
    Return:
        A: symmetric positive matrix LL^T
    """
    L = np.random.rand(n, n)
    for row in range(n):
        for col in range(n):
            if row < col:
                L[row][col] = 0
            if row == col:
                L[row][col] = 5
    A = L @ L.T
    return A

symPosA_10 = genSymPosMatrix(10)
symPosA_100 = genSymPosMatrix(100)

print(np.linalg.norm(symPosA_10, 2)*np.linalg.norm(np.linalg.inv(symPosA_10), 2))
print(np.linalg.norm(symPosA_100, 2)*np.linalg.norm(np.linalg.inv(symPosA_100), 2))


# # Step 2: Generate *b* via matrix-vector product

# In[74]:


x10, x100 = np.random.randint(10, size = (10,)), np.random.randint(100, size = (100,))
b10_actual, b100_actual = np.dot(symPosA_10,x10), np.dot(symPosA_100, x100)


# # Step 3: LU Factorization for A

# In[8]:


def LUFactorization(A: np.ndarray, method: str = 'none'):
    """
    Perform LU factorization of a matrix 3 ways: 
    no pivoting, partial pivoting and complete pivoting
    
    Params: 
        A: a 2d array (matrix) upon which factorization will be performed
        method: method desired denoted as 1, 2, or 3
    
    Return:
        LU: a 2d array (matrix) stored within the array that contained A
        P: Partial pivoting vector (methods 2/3)
        Q: Complete pivoting vector (method 3 only)
    """
    n, m = A.shape
    #ensure we have input a square matrix
    if (n is not m):
        raise Exception("Matrix must be square.")
    
    #in-place LU factorization without pivoting
    def noPivoting(A: np.ndarray, n: int):
        for k in range(n-1):
            if A[k][k] == 0:
                raise ValueError("Null pivot element.")
            A[k+1:, k] = A[k+1:, k] / A[k][k]
            for j in range(k + 1, n):
                for i in range (k + 1, n):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
        return A
    
    #in-place LU factorization with partial pivoting 
    def partialPivoting(A, n):
        pivot = np.arange(0, n)
        for k in range(n-1):
            # pivot rows based on highest value in A
            maxIndex = np.argmax(abs(A[k:,k])) + k
            
            pivot[[k, maxIndex]] = pivot[[maxIndex, k]]
            #swap current row with row with maximum value
            A[[k, maxIndex]] = A[[maxIndex, k]] 
            
            if A[k, k] == 0:
                raise ValueError("Null pivot element.")
                
            A[k+1:, k] = A[k+1:, k] / A[k][k]
            for j in range(k + 1, n):
                for i in range (k + 1, n):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
        return A, pivot
    
    #in-place LU factorization with complete pivoting.
    def completePivoting(A: np.ndarray, n:int):
        #complete pivot based on highest values in A
        rowPivot = np.arange(n)
        colPivot = np.arange(n)
        
        for k in range(n-1):
            #maxIndexRow = np.argmax(abs(A[k:, k:])) // A[k:,k:].shape[0] + k
            #maxIndexCol = np.argmax(abs(A[k:, k:].T)) // A[k:, k:].shape[1] + k
            maxIndex = np.argmax(abs(A[k:, k:]))
            maxIndexRow = maxIndex // A[k:, k:].shape[1] + k
            maxIndexCol = maxIndex % A[k:, k:].shape[1] + k
                          
            if maxIndexRow - k > 0:
                rowPivot[[k, maxIndexRow]] = rowPivot[[maxIndexRow, k]]
                #swap current row with row with max value
                A[[k, maxIndexRow], :] = A[[maxIndexRow, k], :] 
            if maxIndexCol - k > 0:
                colPivot[[k, maxIndexCol]] = colPivot[[maxIndexCol, k]]
                #swap current col with col with max value
                A[:, [k, maxIndexCol]] = A[:, [maxIndexCol, k]]
            
            if A[k][k] == 0:
                raise ValueError("Null pivot element.")
                
            A[k+1:, k] = A[k+1:, k] / A[k][k]
            for j in range(k + 1, n):
                for i in range (k + 1, n):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]
        
        return A, rowPivot, colPivot
    
    if (method == 'none'):    
        return noPivoting(A, n)
        
    if (method == 'partial'):
        return partialPivoting(A, n)
    
    if (method == 'complete'):
        return completePivoting(A, n)


# In[9]:


exMat = np.array([[1, 1, 1], [4, 3, -1], [3, 5, 3]], float)
exMat1 = np.array([[3, 17, 10], [2, 4, -2], [6, 18, -12]], float)
exMat2 = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]], float)

print(LUFactorization(exMat))
print(LUFactorization(exMat1, 'partial'))
print(LUFactorization(exMat2, 'complete'))


# # Step 4: solve Ly = b and Ux = y

# In[64]:


def forwardSub(A: np.ndarray, b: np.ndarray, method: str):
    '''
    Params:
        A: Matrix containing both L and U of shape (n,n)
        b: column vector of shape (n,)
        method: takes 'row' or 'column' for method-orientation
    
    Return:
        b: solution to Ly = b overwritten on b if 'row'
    OR
        y: column vector of solutions in new array if 'col'
    '''
    n, m = A.shape
    L = np.array(A)
    L[range(n), range(n)] = 1
    
    if method == 'row':
        b[0] = b[0] / L[0][0]
        for i in range(1, n):
            b[i] = ((b[i] - L[i][0:i].dot(b[:i]))/ L[i][i])
        return b
    
    if method == 'col':
        y = np.array(b)
        for j in range(n-1):
            y[j] = y[j] / L[j][j]
            y[j + 1:n] = y[j + 1:n] - y[j] * L[j+1:n, j]
        y[n-1] = y[n-1] / L[n-1][n-1]
        return y
    
def backwardSub(A: np.ndarray, b, method: str):
    '''
    Params:
        A: Matrix containing both L and U of shape (n,n)
        b: column vector of shape (n,)
        method: takes 'row' or 'column' for method-orientation
    
    Return:
        b: solution to Ly = b overwritten on b if 'row'
    OR
        x: column vector of solutions in new array if 'col'
    '''
    n, m = A.shape
    
    if method == 'row':
        b[n-1] = b[n-1] / A[n-1][n-1]
        for i in reversed(range(n-1)):
            b[i] = (b[i] - A[i,i+1:].dot(b[i+1:])) / A[i][i]
        return b
    
    if method == 'col':
        x = np.array(b)
        for i in reversed(range(1, n)):
            x[i] = x[i] / A[i][i]
            x[0:i] = x[0:i] - x[i] * A[0:i, i]
        x[0] = x[0] / A[0][0]
        return x

def solver(LU: np.ndarray, b: np.ndarray, ori: str, pivot: str, P: np.ndarray = None, Q: np.ndarray = None):
    '''
    Program to solve Ax = b for x
    Params:
        LU: 2d-array containing L and U
        b: vector stored in a 1d-array
        ori: orientation type ('row' or 'col')
        pivot: pivot type ('none', 'partial', 'complete')
        P: 1d pivot array for partial pivoting
        Q: 1d pivot array for complete pivoting
    Return:
        x: 1d solution vector
    '''
    if pivot != 'none':
        b = b[P].copy()
    y = forwardSub(LU, b, ori)
    x = backwardSub(LU, y, ori)
    if pivot == 'complete':
        x = x[Q]        
    return x


# In[11]:


exMat = np.array([[1, 1, 1], [4, 3, -1], [3, 5, 3]])
LUFactorization(exMat)
bTest = [1., 2, 3]
print(forwardSub(exMat, bTest, 'row'))

bTest = [1., 2, 3]
print(forwardSub(exMat, bTest, 'col'))

bTest = [1., 2, 3]
print(backwardSub(exMat, bTest, 'row'))

bTest = [1., 2, 3]
print(backwardSub(exMat, bTest, 'col'))


# ## Extra Credit

# In[12]:


exMat = np.array([[1, 1, 1], [4, 3, -1], [3, 5, 3]], float)
LUFactorization(exMat)

print('forward sub row-oriented time:')
get_ipython().run_line_magic('timeit', "-n 1000 forwardSub(exMat, [1., 2., 3.], 'row')")

print('\nforward sub col-oriented time:')
get_ipython().run_line_magic('timeit', "-n 1000 forwardSub(exMat, [1., 2., 3.], 'col')")

print('\nbackward sub row-oriented time:')
get_ipython().run_line_magic('timeit', "-n 1000 backwardSub(exMat, [1., 2., 3.], 'row')")

print('\nbackward sub row-oriented time:')
get_ipython().run_line_magic('timeit', "-n 1000 backwardSub(exMat, [1., 2., 3.], 'col')")


# In[13]:


b = np.array([1., 2, 3],float)
A = np.array([[1., 1, 1], [4, 3, -1], [3, 5, 3]])
LUFactorization(A)
x = solver(A, b, 'row', 'none')
print('x =', x)

b1 = np.array([1, 2, 3], float)
A1 = np.array([[1, 1, 1], [4, 3, -1], [3, 5, 3]], 'float')
A1, P1 = LUFactorization(A1, 'partial')
x1 = solver(A1, b1, 'row', 'partial', P1)
print('x1 =',x1)


# In[14]:


b = np.array([1, 2, 3], float)
A = np.array([[3, 17, 10], [2, 4, -2], [6, 18, -12]], float)
A, P = LUFactorization(A, 'partial')
x = solver(A, b, 'row', 'partial', P)
print(x)


# # Step 5: Check the accuracy

# In[15]:


def getLUMatrices(A):
    n, m = A.shape
    def triOnes(n: int, m = None, k = 0):
        #if num cols not specified, make square matrix
        if m is None:
            m = n
        mat = np.greater_equal.outer(np.arange(n), np.arange(-k, m - k))
        return mat

    lowerMask = triOnes(*A.shape[-2:], k = 0)
    L = np.where(lowerMask, A, np.zeros(1))
    #set all diags to 1
    L[range(n), range(n)] = 1
    
    upperMask = triOnes(*A.shape[-2:], k = -1)
    U = np.where(upperMask, np.zeros(1), A)
    return L, U

def getLUProduct(A: np.ndarray):
    L, U = getLUMatrices(A)
    M = L @ U
    return M

def checkFactAcc(A, LU, norm, P = None, Q = None):
    M = getLUProduct(LU)
    PAQ = A
    if P is not None:
        PAQ = PAQ[P, :]
    if Q is not None:
        PAQ = PAQ[:, Q]
    num = np.linalg.norm(PAQ - M, norm)
    denom = np.linalg.norm(A, norm)
    return num / denom

def checkSolAcc(xact, xest, norm):
    num = np.linalg.norm(xact - xest, norm)
    denom = np.linalg.norm(xact, norm)
    return num / denom

def checkResAcc(A, xest, b, norm):
    num = np.linalg.norm(b - (A @ xest), norm)
    denom = np.linalg.norm(b)
    return num / denom


# In[16]:


exMat2 = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]], float)
exLU, exP, exQ = LUFactorization(exMat2, 'complete')
exMat2 = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]], float)
factAcc = checkFactAcc(exMat2, exLU, 'fro', exP, exQ)
print('Factorization accuracy:', factAcc)


# # Correctness Test Task

# In[17]:


A_test = np.array([[2, 1, 0], [-4, 0, 4], [2, 5, 10]], float)
b_test = np.array([3., 0., 17.])
Matrix_p, p = LUFactorization(A_test, 'partial')
print(Matrix_p, p)
x_p = solver(Matrix_p, b_test, 'row', 'partial', p)
print(x_p)

#write results of partial pivoting to txt file in data folder
partTestStr = 'M_p = \n' + str(Matrix_p) + '\nP = ' + str(p) + '\nx_p = ' + str(x_p)
with open(os.path.join(data, 'Partial Pivoting Test.txt'), 'w') as f:
    f.write(partTestStr)

A_test = np.array([[2, 1, 0], [-4, 0, 4], [2, 5, 10]], float)
b_test = np.array([3., 0, 17])
Matrix_c, p_c, q_c = LUFactorization(A_test, 'complete')
print(Matrix_c, p_c, q_c)
x_c = solver(Matrix_c, b_test, 'row', 'complete', p_c, q_c)
print(x_c)

#write results of complete pivoting to txt file in data folder
compTestStr = 'M_c = \n' + str(Matrix_c) + '\nP_c = ' + str(p_c) +'\nQ_c = ' + str(q_c) + '\nx_c = ' + str(x_c)
with open(os.path.join(data, 'Complete Pivoting Test.txt'), 'w') as f:
    f.write(compTestStr)


# In[100]:


#accuracy for partial pivoting factorization
A_test = np.array([[2, 1, 0], [-4, 0, 4], [2, 5, 10]], float)
partFactAcc1 = checkFactAcc(A_test, Matrix_p, 'fro', p)
partFactAcc2 = checkFactAcc(A_test, Matrix_p, 1, p)

#accuracy for complete pivoting factorization 
compFactAcc1 = checkFactAcc(A_test, Matrix_c, 'fro', p_c, q_c)
compFactAcc2 = checkFactAcc(A_test, Matrix_c, 1, p_c, q_c)

#accuracy of solution from partial pivoting
x_test = [1, 1, 1]
partSolAcc1 = checkSolAcc(x_test, x_p, 1)
partSolAcc2 = checkSolAcc(x_test, x_p, 2)

#accuracy of solution from complete pivoting
compSolAcc1 = checkSolAcc(x_test, x_c, 1)
compSolAcc2 = checkSolAcc(x_test, x_c, 2)


#accuracy of residual from partial pivoting
A_test = np.array([[2, 1, 0], [-4, 0, 4], [2, 5, 10]], float)
partResAcc1 = checkResAcc(A_test, x_p, b_test, 1)
partResAcc2 = checkResAcc(A_test, x_p, b_test, 2)

#accuracy of residual from complete pivoting
compResAcc1 = checkResAcc(A_test, x_c, b_test, 1)
compResAcc2 = checkResAcc(A_test, x_c, b_test, 2)


partAccStr1 = '||partFac||_1 = ' + str(partFactAcc1)
partAccStr2 = '||partFac||_F = ' + str(partFactAcc2)
partSolAccStr1 = '||sol||_1 = ' + str(partSolAcc1)
partSolAccStr2 = '||sol||_2 = ' + str(partSolAcc2)
partResAccStr1 = '||res||_1 = ' + str(partResAcc1)
partResAccStr2 = '||res||_2 = ' + str(partResAcc2)
np.savetxt(os.path.join(data, 'Partial Accuracy Tests.txt'), (partAccStr1, partAccStr2, partSolAccStr1, partSolAccStr2, partResAccStr1, partResAccStr2), '%s')

compAccStr1 = '||compFac||_1 = ' + str(compFactAcc1)
compAccStr2 = '||compFac||_F = ' + str(compFactAcc2)
compSolAccStr1 = '||sol||_1 = ' + str(compSolAcc1)
compSolAccStr2 = '||sol||_2 = ' + str(compSolAcc2)
compResAccStr1 = '||res||_1 = ' + str(compResAcc1)
compResAccStr2 = '||res||_2 = ' + str(compResAcc2)
np.savetxt(os.path.join(data, 'Complete Accuracy Tests.txt'), (compAccStr1, compAccStr2, compSolAccStr1, compSolAccStr2, compResAccStr1, compResAccStr2), '%s')


# # Test Original Constructed Matrices

# In[65]:


#Check the accuracy for the 10x10 symmetric matrix created in step 1

symPosA_10_copy = symPosA_10.copy()
symPosA_10_p, P_10 = LUFactorization(symPosA_10_copy, 'partial')
symXp_10 = solver(symPosA_10_p, b10_actual, 'row', 'partial', P_10)

print("Accuracy Checks for 10x10 symmetric A:")
print("Partial pivoting LU factorization values:")
print(checkFactAcc(symPosA_10, symPosA_10_p, 'fro', P_10))
print(checkFactAcc(symPosA_10, symPosA_10_p, 1, P_10))
print(checkSolAcc(x10, symXp_10, 1))
print(checkSolAcc(x10, symXp_10, 2))
print(checkResAcc(symPosA_10, symXp_10, b10_actual, 1))
print(checkResAcc(symPosA_10, symXp_10, b10_actual, 2))


symPosA_10_copy = symPosA_10.copy()
symPosA_10_c, Pc_10, Qc_10 = LUFactorization(symPosA_10_copy, 'complete')
symXc_10 = solver(symPosA_10_c, b10_actual, 'row', 'complete', Pc_10, Qc_10)

print("\nComplete pivoting LU factorization values:")
print(checkFactAcc(symPosA_10, symPosA_10_c, 'fro', Pc_10, Qc_10))
print(checkFactAcc(symPosA_10, symPosA_10_c, 1, Pc_10, Qc_10))
print(checkSolAcc(x10, symXc_10, 1))
print(checkSolAcc(x10, symXc_10, 2))
print(checkResAcc(symPosA_10, symXc_10, b10_actual, 1))
print(checkResAcc(symPosA_10, symXc_10, b10_actual, 2))


# In[67]:


#Check the accuracy for the 100x100 symmetric matrix created in step 1

symPosA_100_copy = symPosA_100.copy()
symPosA_100_p, P_100 = LUFactorization(symPosA_100_copy, 'partial')
symXp_100 = solver(symPosA_100_p, b100_actual, 'row', 'partial', P_100)

print("Accuracy Checks for 100x100 symmetric A:")
print("Partial pivoting LU factorization values:")
print(checkFactAcc(symPosA_100, symPosA_100_p, 'fro', P_100))
print(checkFactAcc(symPosA_100, symPosA_100_p, 1, P_100))
print(checkSolAcc(x100, symXp_100, 1))
print(checkSolAcc(x100, symXp_100, 2))
print(checkResAcc(symPosA_100, symXp_100, b100_actual, 1))
print(checkResAcc(symPosA_100, symXp_100, b100_actual, 2))


symPosA_100_copy = symPosA_100.copy()
symPosA_100_c, Pc_100, Qc_100 = LUFactorization(symPosA_100_copy, 'complete')
symXc_100 = solver(symPosA_100_c, b100_actual, 'row', 'complete', Pc_100, Qc_100)

print("\nComplete pivoting LU factorization values:")
print(checkFactAcc(symPosA_100, symPosA_100_c, 'fro', Pc_100, Qc_100))
print(checkFactAcc(symPosA_100, symPosA_100_c, 1, Pc_100, Qc_100))
print(checkSolAcc(x100, symXc_100, 1))
print(checkSolAcc(x100, symXc_100, 2))
print(checkResAcc(symPosA_100, symXc_100, b100_actual, 1))
print(checkResAcc(symPosA_100, symXc_100, b100_actual, 2))


# In[99]:


#Check the accuracy for the 10x10 random matrix created in step 1

randb10_actual = np.dot(A_10, x10)
randA_10_copy = A_10.copy()
randA_10_p, P_10 = LUFactorization(randA_10_copy, 'partial')
randXp_10 = solver(randA_10_p, randb10_actual, 'row', 'partial', P_10)

print("Accuracy Checks for 10x10 random A:")
print("Partial pivoting LU factorization values:")
print(checkFactAcc(A_10, randA_10_p, 'fro', P_10))
print(checkFactAcc(A_10, randA_10_p, 1, P_10))
print(checkSolAcc(x10, randXp_10, 1))
print(checkSolAcc(x10, randXp_10, 2))
print(checkResAcc(A_10, randXp_10, randb10_actual, 1))
print(checkResAcc(A_10, randXp_10, randb10_actual, 2))


randA_10_copy = A_10.copy()
randA_10_c, Pc_10, Qc_10 = LUFactorization(randA_10_copy, 'complete')
randXc_10 = solver(randA_10_c, randb10_actual, 'row', 'complete', Pc_10, Qc_10)

print("\nComplete pivoting LU factorization values:")
print(checkFactAcc(A_10, randA_10_c, 'fro', Pc_10, Qc_10))
print(checkFactAcc(A_10, randA_10_c, 1, Pc_10, Qc_10))
print(checkSolAcc(x10, randXc_10, 1))
print(checkSolAcc(x10, randXc_10, 2))
print(checkResAcc(A_10, randXc_10, randb10_actual, 1))
print(checkResAcc(A_10, randXc_10, randb10_actual, 2))


# In[98]:


#Check the accuracy for the 100x100 random matrix created in step 1

randb100_actual = np.dot(A_100, x100)
randA_100_copy = A_100.copy()
randA_100_p, P_100 = LUFactorization(randA_100_copy, 'partial')
randXp_100 = solver(randA_100_p, randb100_actual, 'row', 'partial', P_100)

print("Accuracy Checks for 100x100 random A:")
print("Partial pivoting LU factorization values:")
print(checkFactAcc(A_100, randA_100_p, 'fro', P_100))
print(checkFactAcc(A_100, randA_100_p, 1, P_100))
print(checkSolAcc(x100, randXp_100, 1))
print(checkSolAcc(x100, randXp_100, 2))
print(checkResAcc(A_100, randXp_100, randb100_actual, 1))
print(checkResAcc(A_100, randXp_100, randb100_actual, 2))


randA_100_copy = A_100.copy()
randA_100_c, Pc_100, Qc_100 = LUFactorization(randA_100_copy, 'complete')
randXc_100 = solver(randA_100_c, randb100_actual, 'row', 'complete', Pc_100, Qc_100)

print("\nComplete pivoting LU factorization values:")
print(checkFactAcc(A_100, randA_100_c, 'fro', Pc_100, Qc_100))
print(checkFactAcc(A_100, randA_100_c, 1, Pc_100, Qc_100))
print(checkSolAcc(x100, randXc_100, 1))
print(checkSolAcc(x100, randXc_100, 2))
print(checkResAcc(A_100, randXc_100, randb100_actual, 1))
print(checkResAcc(A_100, randXc_100, randb100_actual, 2))

