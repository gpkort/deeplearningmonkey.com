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

    def __init__(self, xy_pairs: dict):
        if len(xy_pairs) < 2:
            raise ValueError("The length xy_pairs may not be zero.")
        if None in xy_pairs.keys() or None in xy_pairs.values():
            raise ValueError("xy_pairs may not contain None")
        if len([i for i in xy_pairs.keys() if type(i) is not float and type(i) is not int]) != 0:
            raise ValueError("Keys in xy_pairs may only contain floats or ints")
        if len([i for i in xy_pairs.values() if type(i) is not float and type(i) is not int]) != 0:
            raise ValueError("Values in xy_pairs may only contain floats or ints")

        self.__xy_pairs = xy_pairs
        self.__xvalues = xy_pairs.keys()
        self.__yvalues = xy_pairs.values()
        self.__n = len(self.__xvalues)
        self.__xmean = Ut.get_mean(self.__xvalues)
        self.__ymean = Ut.get_mean(self.__yvalues)
        self.__x_stddev = Ut.get_std_dev(self.__xvalues)
        self.__y_stddev = Ut.get_std_dev(self.__yvalues)
        self.__corr = self.__get_corr()

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

    def get_slope_intercept(self):
        cv = Ut.get_sum([(x-self.__xmean) * (y-self.__xmean) for x, y in (zip(self.__xvalues, self.__yvalues))])
        sv = Ut.get_sum([(x-self.__xmean)**2 for x in self.__xvalues])
        slope = cv/sv

        intercept = None
        if 0.0 in self.__xvalues:
            print('found zero')
            intercept = self.__xy_pairs[0.0]
        else:
            intercept = -slope * self.__xvalues[0] + self.__yvalues[0]


        return slope, intercept

    def __get_corr(self):
        xy = [x*y for x, y in (zip(self.__xvalues, self.__yvalues))]
        xsquare = [x**2 for x in self.__xvalues]
        ysquare = [y**2 for y in self.__yvalues]
        covariance = (self.__n * Ut.get_sum(xy)) - (Ut.get_sum(self.__xvalues) * Ut.get_sum(self.__yvalues))
        sigx = self.__n*Ut.get_sum(xsquare) - Ut.get_sum(self.__xvalues)**2
        sigy = self.__n * Ut.get_sum(ysquare) - Ut.get_sum(self.__yvalues) ** 2

        return covariance/math.sqrt(sigx * sigy)





