import math
import numpy as np


def cylinder_area(r: float ,h: float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r > 0 and h > 0:
        return 2 * math.pi * r * (r + h)
    else:
        return np.NaN

#Zadanie 2
#print(np.linspace(1, 10, 5))
#print(np.arange(0, 15, 2.5))

def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.

    """
    wektor1 = np.array([1, 1])
    if n <= 0:
        return None
    if isinstance(n, int):
        if n == 1: #1: 1, 2: 1, 3: 2, 4: 3 itd.
            return np.array([1])
        elif n == 2:
            return wektor1
        for i in range(2, n):
            kolejny_element = wektor1[-1] + wektor1[-2]
            wektor1 = np.append(wektor1, kolejny_element)

        return np.reshape(wektor1, (1, n))
    else:
        return None

#print(fib(15))


def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.matrix([[a, 1, -a],
                   [0, 1, 1],
                   [-a, a, 1]])
    Mt = np.transpose(M)
    Mdet = np.linalg.det(M)

    if Mdet == 0:
        Minv = np.NaN
    else:
        Minv = np.linalg.inv(M)

    return Minv, Mt, Mdet

#print(matrix_calculations(5))

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if m < 0 or n < 0:
        return None

    if isinstance(m, int) and isinstance(n, int):
        macierz1 = np.ones((m, n), dtype=int)
        for i in range(m):
            for j in range(n):
                if i > j:
                    macierz1[i][j] = i
                else:
                    macierz1[i][j] = j

        return macierz1

    else:
        return None

#print(custom_matrix(5, 5))