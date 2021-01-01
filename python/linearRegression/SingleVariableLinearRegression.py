import Utilities


class SingleVariableLinearRegression(object):
    """
        Oject that returns weights and bias of of an array of x,y floats

         To use:
         >>> svlr = SingleVariableLinearRegression(xvalues = some numpy array, yvalues= some values of numpy array)
         >>> svlr.std_deviation()
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
        if len(xvalues) == 0 or len(xvalues) != len(xvalues):
            raise ValueError("The length xvalues or yvalues may be None.")

        self._xvalues = xvalues
        self._yvalues = yvalues
        self._xmean = None
        self._ymean = None
        self._x_std_dev = None

    def x_mean(self) -> float:
        if self._xmean is None:
            self._xmean = self.get_mean(self._xvalues)

        return self._xmean

    def y_mean(self) -> float:
        return self._xmean




