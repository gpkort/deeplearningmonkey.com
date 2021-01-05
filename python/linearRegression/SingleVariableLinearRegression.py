import Utilities as Ut
import math


class SingleVariableLinearRegression(object):
    """
        Oject that returns weights and bias of of an array of x,y floats

         To use:
         >>> svlr = SingleVariableLinearRegression(xvalues = some numpy array, yvalues= some values of numpy array)
         >>> svlr.x_std_dev
         >>> svlr.y_std_dev
         >>> svlr.r_squared()
         >>> svlr.corr() #returns Pearson correlation
         >>> svlr.x_mean()
         >>> svlr.y_mean()
         >>> svlr.means
         >>> svlr.weight
         >>> svlr.bias
    """

    def __init__(self, xvalues:list, yvalues:list):
        if xvalues is None or yvalues is None:
            raise ValueError("Neither xvalues or yvalues may be None.")
        if len(xvalues) == 0 or len(yvalues) == 0:
            raise ValueError("The length xvalues or yvalues may not be zero.")
        if len(xvalues) != len(yvalues):
            raise ValueError("The length xvalues and yvalues must be equal.")

        self.__xvalues = xvalues
        self.__yvalues = yvalues
        self.__n = len(self.__xvalues)
        self.__xmean = Ut.get_mean(self.__xvalues)
        self.__ymean = Ut.get_mean(self.__yvalues)
        self.__x_stddev = Ut.get_std_dev(self.__xvalues)
        self.__y_stddev = Ut.get_std_dev(self.__yvalues)
        self.__corr = self.__get_corr()
        self.__intercept = self.__get_intercept()
        self.__slope = self.__get_slope()

    @property
    def x_mean(self) -> float:
        return self.__xmean

    @property
    def y_mean(self) -> float:
        return self.__ymean

    @property
    def x_stddev(self):
        return self.__x_stddev

    @property
    def y_stddev(self):
        return self.__y_stddev

    @property
    def corr(self):
        return self.__corr

    @property
    def get_slope_intercept(self):
        return self.__slope, self.__intercept

    def __get_slope(self):
        return None

    def __get_intercept(self):
        return None

    def __get_corr(self):
        xy = [x*y for x, y in (zip(self.__xvalues, self.__yvalues))]
        xsquare = [x**2 for x in self.__xvalues]
        ysquare = [y**2 for y in self.__yvalues]
        covariance = (self.__n * Ut.get_sum(xy)) - (Ut.get_sum(self.__xvalues) * Ut.get_sum(self.__yvalues))
        sigx = self.__n*Ut.get_sum(xsquare) - Ut.get_sum(self.__xvalues)**2
        sigy = self.__n * Ut.get_sum(ysquare) - Ut.get_sum(self.__yvalues) ** 2

        return covariance/math.sqrt(sigx * sigy)





