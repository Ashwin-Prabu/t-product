import numpy as np
import scipy as sp

def unfold(X):
  """
  Unfolding Tensor by frontal slices.
  M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
  A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
  """
  slices_list = []
  for i in range(0, len(X[0,0])):
    slices_list.append(X[:,:,i])
  X_unf = np.vstack(slices_list)
  return X_unf

def bcirc(X):
  """
  Circulant Matrix based on frontal slices.
  M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
  A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
  """
  num_slices = len(X[0,0])
  list_1 = []
  for i in range(num_slices):
    list_2 = []
    for j in range(num_slices-i, num_slices):
      list_2.append(X[:,:,j])
    for j in range(0, num_slices-i):
      list_2.append(X[:,:,j])
    list_1.append(np.vstack(list_2))
  return np.hstack(list_1)

def fold(X, num_slices):
  """
  Folding Matrix back to tensor by frontal slice.
  M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
  A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
  """
  depth = len(X)//num_slices
  slices_list = []
  for i in range(num_slices):
    slices_list.append(X[depth*i:depth*(i+1),:])
  X_fold = np.stack(slices_list, axis=2)
  return X_fold

def t_product(A, B):
  """
  Implementation of t-product for tensors.
  M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
  A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
  """
  return fold(np.matmul(bcirc(A), unfold(B)), A.shape[2])

def M_hat(M):
  """
  Moving tensor into Fourier domain.
  M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
  A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
  """
  m,l,n = M.shape
  I_m = np.identity(m)
  F_n = np.fft.fft(np.eye(n))*(1/(n**0.5))
  K_1 = np.kron(F_n, I_m)
  I_l = np.identity(l)
  F_nH = F_n.conj().T
  K_2 = np.kron(F_nH, I_l)
  M_hat_diag = np.matmul(K_1, np.matmul(bcirc(M), K_2))
  M_hat_list = []
  for i in range(n):
    M_hat_list.append(M_hat_diag[m*i:(m*i)+m,l*i:(l*i)+l])
  M_hat = np.vstack(M_hat_list)
  M_hat = fold(M_hat, n)
  return M_hat

def identity_tensor(m,n):
  """
  Creates identity tesnor with respect to t-product.
  M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
  A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
  """ 
  front = np.identity(m)
  slice_list = [front]
  for i in range(n-1):
    slice_list.append(np.zeros((m,m)))
  return np.stack(slice_list, axis=2)

def conjT(X):
    """
    Implements conjugate transpose of tensor with respect to t-product.
    M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
    A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
    """
    list_slices = [X[:,:,0].conj().T]
    for i in range(X.shape[2]-1, 0, -1):
        list_slices.append(X[:,:,i].conj().T)
    return np.stack(list_slices, axis = 2)

def inverse_row_slice(A_i):
    """
    Calculates inverse for row slices product with its conjugate transpose (refer to Anna Ma's and Denali Molitor's paper TRK).
    A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
    """
    n = A_i.shape[2]
    F_n = np.fft.fft(np.eye(n))*(1/(n**0.5))
    F_nH = F_n.conj().T
    A_i_h = conjT(A_i)
    D = np.matmul(F_n, np.matmul(bcirc(t_product(A_i, A_i_h)), F_nH))
    diagonal = np.linalg.inv(D).diagonal()
    product_temp = np.matmul(F_nH, diagonal)*(1/(n**0.5))
    product_array = np.reshape(product_temp, (len(product_temp), 1))
    return fold(product_array, n)

def recover_hat(M_hat):
    """
    Recovers tensor from Fourier domain.
    M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
    A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194. 
    """
    m,l,n = M_hat.shape
    M_list = []
    for i in range(n):
        M_list.append(M_hat[:,:,i])
    M_hat_diag = sp.sparse.block_diag(M_list).toarray()
    I_m = np.identity(m)
    F_n = np.fft.fft(np.eye(n))*(1/(n**0.5))
    K_1 = np.kron(F_n, I_m)
    I_l = np.identity(l)
    F_nH = F_n.conj().T
    K_2 = np.kron(F_nH, I_l)
    M_circ = np.matmul(np.linalg.inv(K_1), np.matmul(M_hat_diag, np.linalg.inv(K_2)))
    return fold(M_circ[:,:l], n)