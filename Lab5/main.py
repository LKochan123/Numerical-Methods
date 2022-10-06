import numpy as np


def first_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.

    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów

    return (a,b) - krotka zawierająca współczynniki funkcji linowych"""
    
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        return None
    if x.shape != y.shape: 
        return None
    if len(x.shape) != 1 or len(y.shape) != 1:
        return None
   
    else:
        a = []
        b = []
        for k in range(len(y) - 1): 
            a.append((y[k+1] - y[k])/(x[k+1] - x[k]))
            b.append(y[k] - a[k]*x[k])

        return a, b



def cubic_spline(x: np.ndarray, y: np.ndarray, tol=1e-100):
    """
    Interpolacja splajnów cubicznych

    Returns:
    b współczynnik przy x stopnia 1
    c współczynnik przy x stopnia 2
    d współczynnik przy x stopnia 3
    """

    if not(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        return None

    if x.shape != y.shape:
        return None

    if len(x.shape) != 1 or len(y.shape) != 1:
        return None

    else:
        x = np.array(x)
        y = np.array(y)
        ### check if sorted
        if np.any(np.diff(x) < 0):
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

        size = len(x)
        delta_x = np.diff(x)
        delta_y = np.diff(y)

        ### Get matrix A
        A = np.zeros(shape=(size, size))
        b = np.zeros(shape=(size, 1))
        A[0, 0] = 1
        A[-1, -1] = 1

        for i in range(1, size - 1):
            A[i, i - 1] = delta_x[i - 1]
            A[i, i + 1] = delta_x[i]
            A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])
            ### Get matrix b
            b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])

        ### Solves for c in Ac = b
        print('Jacobi Method Output:')
        c = jacobi(A, b, np.zeros(len(A)), tol=tol, n_iterations=1000)

        ### Solves for d and b
        d = np.zeros(shape=(size - 1, 1))
        b = np.zeros(shape=(size - 1, 1))
        for i in range(0, len(d)):
            d[i] = (c[i + 1] - c[i]) / (3 * delta_x[i])
            b[i] = (delta_y[i] / delta_x[i]) - (delta_x[i] / 3) * (2 * c[i] + c[i + 1])

        return b.squeeze(), c.squeeze(), d.squeeze()

def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Iteracyjne rozwiązanie równania Ax=b dla zadanego x0

    Returns:
    x - estymowane rozwiązanie
    """
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol + 1

    while (x_diff > tol) and (counter < n_iterations):  # iteration level
        for i in range(0, n):  # element wise level for x
            s = 0
            for j in range(0, n):  # summation for i !=j
                if i != j:
                    s += A[i, j] * x_prev[j]

            x[i] = (b[i] - s) / A[i, i]
        # update values
        counter += 1
        x_diff = (np.sum((x - x_prev) ** 2)) ** 0.5
        x_prev = x.copy()  # use new x for next iteration

    print("Number of Iterations: ", counter)
    print("Norm of Difference: ", x_diff)
    return x


