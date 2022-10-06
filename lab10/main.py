import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle
from numpy import linalg as LA

# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    
    if isinstance(x, np.ndarray):
        return P.polyfromroots(x)

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if isinstance(a, np.ndarray):
        for i in range(len(a)):
            a[i] += 1e-10 * np.random.random_sample()

        return a, P.polyroots(a)


# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if isinstance(wsp, np.ndarray):
        n = len(wsp)
        frobenious_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j-1:
                    frobenious_matrix[i][j] = 1
                if i == n-1:
                    frobenious_matrix[i][j] = -1 * wsp[j]

        eig_vector = LA.eigvals(frobenious_matrix)
        schure_vector = scipy.linalg.schur(frobenious_matrix)
        x0_vector = P.polyroots(wsp)

        return frobenious_matrix, eig_vector, schure_vector, x0_vector





