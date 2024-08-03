The following package contains the different tools required in implementing the t-product. 

# 1. Requirements
1) Numpy
2) Scipy

# 2. Functions

## Unfold
    Unfolding Tensor by frontal slices.

## Bcirc
    Circulant Matrix based on frontal slices.

## Fold
    Folding Matrix back to tensor by frontal slice.

## T-Product
    Implementation of t-product for tensors.

## M_hat
    Moving tensor into Fourier domain.

## Identity Tesnor
    Creates identity tesnor with respect to t-product.

## ConjT
    Implements conjugate transpose of tensor with respect to t-product.

## Inverse Row Slices
    Calculates inverse for row slices product with its conjugate transpose (refer to [Anna Ma's and Denali Molitor's paper TRK](https://arxiv.org/pdf/2006.01246)).

## Recover Hat
    Recovers tensor from Fourier domain. 

# 3. Acknowledgements
    This package is result of the theoretical results derived from two papers: ["Factorization strategies for third-order tensors": by Misha E. Kilmer, and Carla D. Martin](https://www.sciencedirect.com/science/article/pii/S0024379510004830), and ["Randomized Kaczmarz for Tensor Linear Systems" by Anna Ma, and Denali Molitor]. Further theoretical results are present in both papers. 

# 4. References
    1. M. E. Kilmer and C. D. Martin. Factorization strategies for third-order tensors. Linear Algebra App., 435(3):641–658, 2011.
    2. A. Ma, D. Molitor, Randomized Kaczmarz for tensor linear systems, BIT Numer.Math. 62 (2022) 171–194.
