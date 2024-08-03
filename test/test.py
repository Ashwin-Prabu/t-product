from    # The code to test
import unittest
import numpy as np

class Test_T_Product_Operations(unittest.TestCase):
    A = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    B = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    C = np.array([[[ 44,  42],[ 64,  62]],[[100,  98],[152, 150]]])

    A_hat = np.array([[[ 3.+0.j, -1.+0.j],[ 7.+0.j, -1.+0.j]],[[11.+0.j, -1.+0.j],[15.+0.j, -1.+0.j]]])
    A_u = np.array([[1, 3],[5, 7],[2, 4],[6, 8]])
    A_b = np.array([[1, 3, 2, 4],[5, 7, 6, 8],[2, 4, 1, 3],[6, 8, 5, 7]])
    A_t = np.array([[[1, 2],[5, 6]],[[3, 4],[7, 8]]])

    def test_unfold(self):
        temp = t_product_functions.unfold(self.A)
        self.assertTrue(temp - self.A_u == 0)

    def test_unfold(self):
        temp = t_product_functions.bcirc(self.A)
        self.assertTrue(temp - self.A_u == 0)

    def test_fold(self):
        temp = t_product_functions.fold(self.A_u, 2)
        self.assertTrue(temp - self.A == 0)

    def test_t_product(self):
        temp = t_product_functions.t_product(self.A, self.B)
        self.assertTrue(temp - self.C <= 10**-10)


    def test_M_hat(self):
        temp = t_product_functions.M_hat(self.A)
        self.assertTrue(temp - self.A_hat  <= 10**-10)


    def test_identity_tensor(self):
        temp1 = np.array([[[1., 0.],[0., 0.]],[[0., 0.],[1., 0.]]])
        temp2 = t_product_functions.identity_tensor(2,2)
        self.assertTrue(temp1 == temp2)


    def test_conjT(self):
        temp = t_product_functions.conjT(self.A)
        self.assertTrue(self.A_t == temp)

    def test_inverse_row_slices(self):
        temp = t_product_functions.inverse_row_slice(self.A[0:1,:,:])
        A_i = t_product_functions.t_product(self.A[0:1,:,:], t_product_functions.conjT(self.A[0:1,:,:]))
        I = t_product_functions.identity_tensor(1,2)
        self.assertTrue(t_product_functions.t_product(A_i, t_product_functions.inverse_row_slice(self.A[0:1,:,:]))-I <= 10**-10)

    def test_recover_hat(self):
        temp = t_product_functions.recover_hat(self.A_hat)
        self.assertTrue(temp - self.A <= 10**-10)

if __name__ == '__main__':
    unittest.main()
