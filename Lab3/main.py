import numpy as np
import scipy
import pickle
import math

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int, float, list, np.ndarray)) and isinstance(v_aprox, (int, float, list, np.ndarray)):
        if isinstance(v, (int, float)) and isinstance(v_aprox, (int, float)):
            return np.abs(v - v_aprox)
        if isinstance(v, list) and isinstance(v_aprox, list):
            v = np.array(v)
            if len(v) == len(v_aprox):
                return np.abs(v-v_aprox)
            else:
                return np.nan
        if isinstance(v, (int, float)) and isinstance(v_aprox, list):
            lista1 = np.zeros(len(v_aprox))
            for i in range(len(v_aprox)):
                lista1[i] = np.abs(v - v_aprox[i])
            return lista1
        if isinstance(v, (int, float)) and isinstance(v_aprox, np.ndarray) or isinstance(v, np.ndarray) and isinstance(v_aprox, (int, float)):
            return np.abs(v-v_aprox)
        if isinstance(v, np.ndarray) and isinstance(v_aprox, np.ndarray):
            if all((m == n) or (m == 1) or (n == 1) for m, n in zip(v.shape[::-1], v_aprox.shape[::-1])):
                return np.abs(v - v_aprox)
            else:
                return np.NaN

    else:
        return np.nan


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    b_bezw = absolut_error(v, v_aprox)
    if b_bezw is np.nan:
        return np.nan

    if isinstance(v, np.ndarray):
        return b_bezw/v

    if isinstance(v, (int, float)):
        if v == 0:
            return np.nan

    if isinstance(v, np.ndarray) and not v.any():
        return np.nan

    if isinstance(v, list) and isinstance(v_aprox, np.ndarray):
        for i in range(len(v)):
            if v[i] == 0:
                return np.nan
    else:
        return b_bezw/ v


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) and isinstance(c, (int, float)):
        b = 2 ** n
        p1 = b - b + c
        p2 = b + c - b
        return np.abs(p1 - p2)
    else:
        return np.NaN


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(x, (int, float)) and isinstance(n, int):
        e = 0
        if n < 0: #w szeregu Taylora możemy rozwijać od 0 lub większej liczby całkowitej
            return np.NaN
        for i in range(n):
            e = e + (x**i)/(scipy.math.factorial(i))
        return e


    else:
        return np.NaN


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) and isinstance(x, (int, float)):
        m = k - 1
        if k < 0:
            return np.NaN
        if k == 0:
            return 1
        if k == 1:
            return np.cos(x)
        else:
            return 2*np.cos(x) * np.cos(m*x) - np.cos(m*x-x)
    else:
        return np.NaN


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) and isinstance(x, (int, float)):
        m = k
        if k < 0:
            return np.nan
        if k == 0:
            return 1, 0 #cos, sin
        if k == 1:
            return np.cos(x), np.sin(x)
        else:
            return np.cos(x) * np.cos(m*x-x) - np.sin(x) * np.sin(m*x - x), np.sin(x) * np.cos(m*x-x) + np.cos(x) * np.sin(m*x-x)
    else:
        return np.NaN


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int):
        if n < 1:
            return np.NaN
        k = 0
        for i in range(1, n+1):
            k += 1/(i**2)
        return math.sqrt(6*k)
    else:
        return np.NaN