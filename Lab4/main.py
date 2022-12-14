##
import numpy as np
from typing import Union, List

def chebyshev_nodes(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n > 0: 
        wezly = np.zeros(n+1)
        for k in range(n + 1):
            wezly[k] = np.cos((k*np.pi)/n)
        return wezly
    else: 
        return None

    
def bar_czeb_weights(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n > 0: 
        wagi_barycentryczne = np.zeros(n+1)
        for i in range(n+1): 
            if i == 0 or i == n:
                omega = 0.5 
                wagi_barycentryczne[i] = np.power(-1, i) * omega
            else:
                wagi_barycentryczne[i] = np.power(-1, i)
        return wagi_barycentryczne
    else:
        return None
    
def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    Y = []
    if all(isinstance(k, np.ndarray) for k in [xi, yi, wi, x]):
        if xi.shape == yi.shape and yi.shape == wi.shape:
            for x in np.nditer(x):
                L = wi/(x - xi)
                Y.append(yi@L/sum(L))

            Y = np.array(Y)
            return Y
    else:
        return None

    
def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(xr, (int, float)) and isinstance(x, (int, float)):
        return np.abs(xr - x)

    if all(isinstance(i, np.ndarray) for i in [xr, x]):
        if np.size(xr) == np.size(x):
            return max(xr - x)
        else:
            return np.NaN

    if all(isinstance(j, List) for j in [xr, x]):
        if len(xr) == len(x):
            return np.abs(max(xr) - max(x))
        else:
            return np.NaN
    else:
        return np.NaN
