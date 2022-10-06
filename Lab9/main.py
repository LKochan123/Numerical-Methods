import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) and \
            isinstance(iteration, int) and iteration > 0:
        if f(a) * f(b) < 0:
            n = 1
            while n <= iteration:
                c = (a + b)/2
                if np.abs(f(c)) < epsilon or np.abs(c) < epsilon:
                    return c, n - 1

                elif f(c)*f(a) < 0:
                    a = a
                    b = c

                elif f(b) * f(c) < 0:
                    a = c
                    b = b
                n += 1

            return (a + b)/2, n




            


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) \
        and isinstance(iteration, int) and iteration > 0 and epsilon > 0:
        if f(a) * f(b) < 0:
            n = 1
            while n <= iteration:
                c = (f(b) * a - f(a) * b) / (f(b) - f(a))

                if f(c)*f(a) <= 0:
                    b = c
                else:
                    a = c

                if np.abs(f(c)) < epsilon or np.abs(b-a) < epsilon:
                    return c, n - 1

                n += 1

            return (f(b) * a - f(a) * b) / (f(b) - f(a)), iteration


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if all(isinstance(i, (int, float)) for i in [a, b]) and all(callable(i) for i in [f, df, ddf]) and \
            isinstance(epsilon, float) and isinstance(iteration, int):

        a_b = np.linspace(a, b, 1000)
        df_val = df(a_b)
        ddf_val = ddf(a_b)
        if not ((np.all(np.sign(df_val) < 0) or np.all(np.sign(df_val) > 0)) and
                (np.all(np.sign(ddf_val) < 0) or np.all(np.sign(ddf_val) > 0))):
            return None

        if f(a) * ddf(a) > 0:
            x = a
        else:
            x = b

        if f(a) * f(b) < 0:
            for i in range(iteration):
                x_n = x - f(x) / df(x)

                if np.abs(x_n - x) < epsilon:
                    return x_n, i

                if i == iteration - 1:
                    return x_n, iteration

                x = x_n
