from unittest import TestCase
import numpy as np
import linearRegression.SingleVariableLinearRegression as svlr


class SingleVariableLinearRegressionTests(TestCase):

    def test_mean(self):
        lr1 = svlr.SingleVariableLinearRegression({1: 4, 2: 8})
        lr2 = svlr.SingleVariableLinearRegression({1: 2, 2: 4, 3: 6})
        lr3 = svlr.SingleVariableLinearRegression({9: 18, 10: 20})
        seqx = (lr1.x_mean, lr2.x_mean, lr3.x_mean)
        seqy = (lr1.y_mean, lr2.y_mean, lr3.y_mean)

        self.assertSequenceEqual(seqx, (1.5, 2, 9.5))
        self.assertSequenceEqual(seqy, (6, 4, 19))

    def test_std_dev(self):
        mu, sigma = 0, 0.1  # mean and standard deviation
        x = np.random.normal(mu, sigma, 1000)
        y = np.array(2 * x)
        xy = {k: v for k, v in zip(x.tolist(), y.tolist())}

        lr1 = svlr.SingleVariableLinearRegression(xy)
        self.assertAlmostEqual(np.std(x), lr1.x_stddev)
        self.assertAlmostEqual(np.std(y), lr1.y_stddev)

    def test_coor(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array(2 * x)
        noise = np.random.normal(0, .2, y.shape)
        new_y = np.array(y + noise)
        xy = {k: v for k, v in zip(x.tolist(), new_y.tolist())}
        lr1 = svlr.SingleVariableLinearRegression(xy)
        coefmatrix = np.corrcoef(x=x, y=new_y)
        self.assertAlmostEqual(coefmatrix[0][1], lr1.corr)

    def test_intercept(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array(2 * x + 5)
        xy = {k: v for k, v in zip(x.tolist(), y.tolist())}

        lr1 = svlr.SingleVariableLinearRegression(xy)
        slope, intercept = lr1.get_slope_intercept()
        self.assertEqual(2, slope)
        self.assertEqual(5, intercept)




