"""
Scripted By: Lars Kühmichel
License: MIT License
GitHub: https://github.com/LarsKue/UncertainValue
"""

import numpy as np
import math


def square(a):
    return a * a


def magnitude(x):
    return int(math.floor(math.log10(abs(x))))


def significant_digit(x):
    return int(x // (10 ** math.floor(math.log10(x))))


class UncertainValue:
    __value = 0
    __uncertainty = 0
    __printprecision = 6
    __automaticprintprecision = True
    __enable_mean_instantiation = True

    def __init__(self, *args):
        if self.__enable_mean_instantiation and (len(args) > 2 or type(args) is list):
            # instantiation of average value and mean error of the mean from measurements
            self.__value = float(sum(args)) / len(args)
            self.__uncertainty = np.sqrt(
                1.0 / (len(args) * (len(args) - 1)) * float(sum(square(val - self.__value) for val in args)))
            return
        if len(args) == 1:
            self.__value = float(args[0])
            return
        if len(args) == 2:
            self.__value = float(args[0])
            self.__uncertainty = float(abs(args[1]))

    @staticmethod
    def sum(valuelist):
        result = valuelist[0]
        for i in range(1, len(valuelist)):
            result += valuelist[i]
        return result

    @staticmethod
    def average(valuelist):
        s = UncertainValue.sum(valuelist)
        return s / len(valuelist)

    @staticmethod
    def np_abs_u(valuelist):
        # intended for use on numpy arrays
        uncertainties = [x.absolute_u() for x in valuelist]
        return np.array(uncertainties)

    @staticmethod
    def np_rel_u(valuelist):
        # intended for use on numpy arrays
        uncertainties = [x.relative_u() for x in valuelist]
        return np.array(uncertainties)

    @staticmethod
    def np_float(valuelist):
        # intended for use on numpy arrays
        values = [float(x) for x in valuelist]
        return np.array(values)

    @staticmethod
    def sin(x):
        return UncertainValue(np.sin(x.value()), x.absolute_u() * np.cos(x.value()))

    @staticmethod
    def cos(x):
        return UncertainValue(np.cos(x.value()), x.absolute_u() * np.sin(x.value()))

    @staticmethod
    def tan(x):
        return UncertainValue.sin(x) / UncertainValue.cos(x)

    @staticmethod
    def arcsin(x):
        return UncertainValue(np.arcsin(x.value()), x.absolute_u() / np.sqrt(1 - square(x.value())))

    @staticmethod
    def arccos(x):
        return UncertainValue(np.arccos(x.value()), x.absolute_u() / np.sqrt(1 - square(x.value())))

    @staticmethod
    def arctan(x):
        return UncertainValue(np.arctan(x.value()), x.absolute_u() / (square(x.value()) + 1))

    @staticmethod
    def sqrt(x):
        return UncertainValue(np.sqrt(x.value()), x.absolute_u() / (2 * np.sqrt(x.value())))

    @staticmethod
    def log(x):
        return UncertainValue(np.log(x.value()), x.absolute_u() / x.value())

    def value(self):
        return self.__value

    def absolute_u(self):
        return self.__uncertainty

    def relative_u(self):
        return self.__uncertainty / self.__value

    def is_exact(self):
        return self.__uncertainty == 0

    def get_print_precision(self):
        return self.__printprecision

    def set_print_precision(self, val):
        self.__printprecision = val

    def set_automatic_print_precision(self, val):
        self.__automaticprintprecision = val

    def set_enable_mean_instantiation(self, val):
        self.__enable_mean_instantiation = val

    def __abs__(self):
        return UncertainValue(abs(self.__value), self.__uncertainty)

    def __add__(self, other):  # +
        if isinstance(other, (int, float)):
            return UncertainValue(self.__value + other, self.__uncertainty)
        return UncertainValue(self.__value + other.value(),
                              np.sqrt(square(self.__uncertainty) + square(other.absolute_u())))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):  # -
        if isinstance(other, (int, float)):
            return UncertainValue(self.__value - other, self.__uncertainty)
        return UncertainValue(self.__value - other.value(),
                              np.sqrt(square(self.__uncertainty) + square(other.absolute_u())))

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):  # *
        if isinstance(other, (int, float)):
            return UncertainValue(self.__value * other, self.__uncertainty * abs(other))
        return UncertainValue(self.__value * other.value(), np.sqrt(
            square(self.__uncertainty * other.value()) + square(self.__value * other.absolute_u())))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):  # / (floating point)
        if isinstance(other, (int, float)):
            return UncertainValue(self.__value / other, self.__uncertainty / abs(other))
        return UncertainValue(self.__value / other.value(), self.__value / other.value() * np.sqrt(
            square(self.relative_u()) + square(other.relative_u())))

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return UncertainValue(other / self.__value, abs(other) * self.__uncertainty / square(self.__value))
        # should technically never be reached
        raise Exception("UncertainValue encountered invalid division operation")
        # use this code if this exception is ever thrown on a valid uv-uv division
        # return UncertainValue(other.value() / self.__value, np.sqrt(square(other.absolute_u() / self.__value) + square(self.__uncertainty * other.value() / square(self.__value))))

    def __iadd__(self, other):  # +=
        if isinstance(other, (int, float)):
            self.__value += other
            return self
        self.__value += other.value()
        self.__uncertainty = np.sqrt(square(self.__uncertainty) + square(other.absolute_u()))
        return self

    def __isub__(self, other):  # -=
        if isinstance(other, (int, float)):
            self.__value -= other
            return self
        self.__value -= other.value()
        self.__uncertainty = np.sqrt(square(self.__uncertainty) + square(other.absolute_u()))
        return self

    def __imul__(self, other):  # *=
        if isinstance(other, (int, float)):
            self.__value *= other
            return self
        self.__value *= other.value()
        self.__uncertainty = np.sqrt(
            square(self.__uncertainty * other.value()) + square(self.__value * other.absolute_u()))
        return self

    def __itruediv__(self, other):  # /= (floating point)
        if isinstance(other, (int, float)):
            self.__value /= other
            return self
        self.__value /= other.value()
        self.__uncertainty = self.__value / other.value() * np.sqrt(
            square(self.relative_u()) + square(other.relative_u()))
        return self

    def __pow__(self, power, modulo=None):
        if isinstance(power, (int, float)):
            return UncertainValue(self.__value ** power, self.__uncertainty * power * self.__value ** (power - 1))
        return UncertainValue(self.__value ** power.value(), np.sqrt(
            (self.__uncertainty * power.value() * self.__value ** (power.value() - 1)) ** 2 + (
                        power.absolute_u() * np.log(power.value()) * self.__value ** power.value()) ** 2))

    """
    DISCLAIMER:
    Uncertain Value Comparison is done by value only, and is only implemented for easy plotting with pyplot.
    These comparison operators do not have scientific meaning behind them, as they ignore uncertainties.    
    """

    def __le__(self, other):
        return self.__value <= other.__value

    def __ge__(self, other):
        return self.__value >= other.__value

    def __lt__(self, other):
        return self.__value < other.__value

    def __gt__(self, other):
        return self.__value > other.__value

    def __float__(self):
        return float(self.__value)

    def __int__(self):
        return int(self.__value)

    # equality operator is not defined for now

    def __str__(self):  # print overload
        """
        With automaticprintprecision set to True, this will return an appropriate representation
        of the uncertain number with its error in the form of (value +/- error) * 10^magnitude.

        When the first significant digit of the error is a 1 or 2, two digits are to be shown
        in this string. Otherwise, only one.

        You may manually set how many digits are to be printed by setting automaticprintprecision to False
        and then altering the printprecision.
        """
        if self.__automaticprintprecision:
            result = "("
            magnitude_value = magnitude(self.__value)
            magnitude_err = magnitude(self.__uncertainty)
            two_digits = significant_digit(self.__uncertainty) == 1 or significant_digit(self.__uncertainty) == 2
            if (magnitude_err >= magnitude_value and not two_digits) or magnitude_err > magnitude_value:
                result += "{value:.0f} \u00b1 {err:.0f}) * 10^{magnitude:d}".format(
                    value=self.__value * 10 ** -magnitude_value, err=self.__uncertainty * 10 ** -magnitude_value,
                    magnitude=magnitude_value)
                return result
            if magnitude_err == magnitude_value and two_digits:
                result += "{value:.1f} \u00b1 {err:.1f}) * 10^{magnitude:d}".format(
                    value=self.__value * 10 ** -magnitude_value, err=self.__uncertainty * 10 ** -magnitude_value,
                    magnitude=magnitude_value)
                return result
            result += "{val_mag_err} \u00b1 {err_mag_err}) * 10^{magnitude:d}".format(
                val_mag_err="{value:." + str(abs(magnitude_err - magnitude_value) + two_digits) + "f}",
                err_mag_err="{err:." + str(abs(magnitude_err - magnitude_value) + two_digits) + "f}",
                magnitude=magnitude_value).format(value=self.__value * 10 ** -magnitude_value,
                                                  err=self.__uncertainty * 10 ** -magnitude_value)
            return result

        return ("({0:." + str(self.__printprecision) + "f} \u00b1 {1:." + str(self.__printprecision) + "f})").format(
            self.__value, self.__uncertainty)
