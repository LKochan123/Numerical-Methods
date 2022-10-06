import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.zeros(shape=(m, m))
        b = np.random.randint(0, 9, size = (m,))
        for i in range(m):
            for j in range(m):
                if i != j:
                    A[i][j] = np.random.randint(0, 2)
                else:
                    A[i][j] = 3000*m*np.random.randint(7, 9)
        return A, b
    else:
        return None


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray) and len(A.shape) > 1 and A.shape[0] == A.shape[1] and A.shape[0] != 0:
        return all((2 * np.abs(np.diag(A))) >= sum(np.abs(A), 1))
    else:
        return None

def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.zeros(shape=(m, m))
        b = np.random.randint(0, 9, size=(m, ))
        for i in range(m):
            for j in range(m):
                if j >= j:
                    A[i][j] = np.random.randint(0, 9)

        for r in range(m):
            for k in range(m):
                if k < r:
                    A[k][r] = A[r][k]
        return A, b

    else:
        return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray) and len(A.shape) > 1 and A.shape[0] == A.shape[1]:
        return np.allclose(A, A.T)

    else:
        return None


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if all(isinstance(i, np.ndarray) for i in [A, b, x_init]) and isinstance(epsilon, float) and isinstance(maxiter, int):
        if len(A.shape) > 1 and A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0] == x_init.shape[0]:
            D = np.diag(np.diag(A)) #macierz diagonalna
            LU = A - D
            m = A.shape[0] #wymiar macierzy
            D1 = np.zeros(shape=(m, m))

            for row in range(m):
                for col in range(m):
                    if row == col:
                        D1[row][col] = 1/D[row][col] #Macierz D^(-1)

            for i in range(maxiter):
                x_init = np.dot(D1, b - np.dot(LU, x_init))

            return x_init, maxiter

    else:
        return None