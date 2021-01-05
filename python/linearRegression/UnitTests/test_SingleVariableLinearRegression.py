from unittest import TestCase
import numpy as np
import linearRegression.SingleVariableLinearRegression as svlr


class SingleVariableLinearRegressionTests(TestCase):

    def test_mean(self):
        lr1 = svlr.SingleVariableLinearRegression(xvalues=[1], yvalues=[1])
        lr2 = svlr.SingleVariableLinearRegression(xvalues=[1, 2, 3], yvalues=[1, 2, 3])
        lr3 = svlr.SingleVariableLinearRegression(xvalues=[9, 10], yvalues=[9, 10])
        seqx = (lr1.x_mean, lr2.x_mean, lr3.x_mean)
        seqy = (lr1.y_mean, lr2.y_mean, lr3.y_mean)
        seq2 = (1, 2, 9.5)
        self.assertSequenceEqual(seqx, seq2)
        self.assertSequenceEqual(seqy, seq2)

    def test_x_mean_nonnumeric(self):
        lr1 = svlr.SingleVariableLinearRegression(xvalues=['a', 'b', 1, 2, 3], yvalues=[1, 2, 'c', 4, 5])
        self.assertEqual(None, lr1.x_mean)
        self.assertEqual(None, lr1.y_mean)

    def test_std_dev(self):
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)
        lr1 = svlr.SingleVariableLinearRegression(yvalues=s.tolist(), xvalues=s.tolist())
        self.assertAlmostEqual(np.std(s), lr1.y_stddev)
        self.assertAlmostEqual(np.std(s), lr1.x_stddev)

    def test_coor(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x
        noise = np.random.normal(0, .2, y.shape)
        new_y = y + noise
        lr1 = svlr.SingleVariableLinearRegression(yvalues=new_y.tolist(), xvalues=x.tolist())
        coefmatrix = np.corrcoef(x=x, y=new_y)
        self.assertAlmostEqual(coefmatrix[0][1], lr1.corr)



