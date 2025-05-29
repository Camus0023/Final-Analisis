import numpy as np
import os

def spectral_radius(matrix):
    return max(abs(np.linalg.eigvals(matrix)))

def format_result(nombre, N, xi, E, tol):
    cabecera = f">> {nombre}\n\nIteración    "
    cabecera += "  ".join([f"x{i+1: <10}" for i in range(len(xi[0]))]) + "  Error"
    sep = "-" * (13 + len(xi[0]) * 12 + 8)
    filas = "\n".join([f"{N[i]: <11}" + "  ".join(f"{val: <12.6f}" for val in xi[i]) + f"  {E[i]:.6f}" for i in range(len(N))])
    final = f"\n\nConvergió en {len(N)} iteraciones con tolerancia = {tol:.6f}"
    return f"{cabecera}\n{sep}\n{filas}{final}\n"

def jacobi_method(A, b, x0, tol, niter, error_type):
    A, b, x0 = np.array(eval(A)), np.array(eval(b)), np.array(eval(x0))
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D) @ (L + U)
    C = np.linalg.inv(D) @ b
    x = x0
    err = tol + 1
    xi, E, N = [], [], []
    c = 0
    while err > tol and c < niter:
        x1 = T @ x + C
        err = np.linalg.norm((x1 - x) / x1, np.inf) if error_type != 'Error Absoluto' else np.linalg.norm(x1 - x, np.inf)
        x = x1
        xi.append(x.copy())
        E.append(err)
        N.append(c+1)
        c += 1
    radio = spectral_radius(T)
    return xi[-1] if err < tol else "No converge", N, xi, E, radio

def gauss_seidel_method(A, b, x0, tol, niter, error_type):
    A, b, x0 = np.array(eval(A)), np.array(eval(b)), np.array(eval(x0))
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D - L) @ U
    C = np.linalg.inv(D - L) @ b
    x = x0
    err = tol + 1
    xi, E, N = [], [], []
    c = 0
    while err > tol and c < niter:
        x1 = T @ x + C
        err = np.linalg.norm((x1 - x) / x1, np.inf) if error_type != 'Error Absoluto' else np.linalg.norm(x1 - x, np.inf)
        x = x1
        xi.append(x.copy())
        E.append(err)
        N.append(c+1)
        c += 1
    radio = spectral_radius(T)
    return xi[-1] if err < tol else "No converge", N, xi, E, radio

def sor_method(A, b, x0, tol, niter, w, error_type):
    A, b, x0 = np.array(eval(A)), np.array(eval(b)), np.array(eval(x0))
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D - w*L) @ ((1 - w)*D + w*U)
    C = w * np.linalg.inv(D - w*L) @ b
    x = x0
    err = tol + 1
    xi, E, N = [], [], []
    c = 0
    while err > tol and c < niter:
        x1 = T @ x + C
        err = np.linalg.norm((x1 - x) / x1, np.inf) if error_type != 'Error Absoluto' else np.linalg.norm(x1 - x, np.inf)
        x = x1
        xi.append(x.copy())
        E.append(err)
        N.append(c+1)
        c += 1
    radio = spectral_radius(T)
    return xi[-1] if err < tol else "No converge", N, xi, E, radio

def generar_informe(resultados):
    mejor = min(
        [(metodo, nombre) for nombre, metodo in resultados.items() if isinstance(metodo[0], np.ndarray)],
        key=lambda x: len(x[0][1]) if isinstance(x[0][1], list) else float('inf'),
        default=(None, None)
    )
    return mejor[1] if mejor[1] else "Ninguno"
