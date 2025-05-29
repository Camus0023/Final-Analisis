import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sympy import symbols, simplify, expand

def vandermonde(xs, ys, x):
    A = np.vander(xs, increasing=True)
    coef = np.linalg.solve(A, ys)
    return np.polyval(coef[::-1], x)

def newton(xs, ys, x):
    n = len(xs)
    coef = np.copy(ys)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xs[j:n] - xs[j-1])
    result = coef[0]
    for i in range(1, n):
        result += coef[i] * np.prod([x - xs[j] for j in range(i)])
    return result

def lagrange(xs, ys, x):
    total = 0
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        total += term
    return total

def spline_lineal(xs, ys, x):
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i+1]:
            return ys[i] + (ys[i+1] - ys[i]) * (x - xs[i]) / (xs[i+1] - xs[i])
    return None

def spline_cubico(xs, ys, x):
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(xs, ys)
        result = float(cs(x))  # Convertir a float para asegurar serialización
        return result
    except ImportError:
        # Fallback simple cubic interpolation
        return lagrange(xs, ys, x)

def construir_polinomio(xs, ys, metodo):
    x = symbols('x')

    if metodo == "Vandermonde":
        coef = np.linalg.solve(np.vander(xs, increasing=True), ys)
        poly = sum(coef[i]*x**i for i in range(len(coef)))

    elif metodo == "Newton":
        n = len(xs)
        coef = np.copy(ys)
        for j in range(1, n):
            coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xs[j:n] - xs[j-1])
        poly = coef[0]
        for i in range(1, n):
            prod = 1
            for j in range(i):
                prod *= (x - xs[j])
            poly += coef[i]*prod

    elif metodo == "Lagrange":
        n = len(xs)
        poly = 0
        for i in range(n):
            term = ys[i]
            for j in range(n):
                if i != j:
                    term *= (x - xs[j]) / (xs[i] - xs[j])
            poly += term

    return str(expand(simplify(poly)))

def generar_grafico(xs, ys, x_eval, metodos):
    plt.figure(figsize=(10, 6))
    x_vals = np.linspace(min(xs)-1, max(xs)+1, 400)
    plt.scatter(xs, ys, color='black', s=50, label='Datos', zorder=5)

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (nombre, metodo) in enumerate(metodos):
        try:
            y_vals = [metodo(xs, ys, xi) for xi in x_vals]
            color = colors[i % len(colors)]
            plt.plot(x_vals, y_vals, label=nombre, color=color, linewidth=2)
        except:
            continue

    plt.axvline(x=x_eval, color='gray', linestyle='--', alpha=0.7, label=f'x_eval = {x_eval}')
    plt.legend()
    plt.title('Métodos de Interpolación')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return img_b64

def evaluar_metodos(xs, ys, x_eval, y_real=None, metodos=[]):
    resultados = []
    polinomios = {}
    real_value = y_real if y_real is not None else float(np.interp(x_eval, xs, ys))

    for nombre, metodo in metodos:
        try:
            y = metodo(xs, ys, x_eval)
            # Convertir a tipo Python nativo para asegurar serialización JSON
            if y is not None:
                if isinstance(y, np.ndarray):
                    y = float(y.item())
                else:
                    y = float(y)
                err = abs(y - real_value)
                resultados.append({"metodo": nombre, "resultado": y, "error": err})
            else:
                resultados.append({"metodo": nombre, "resultado": float('nan'), "error": float('inf')})

            if nombre in ["Vandermonde", "Newton", "Lagrange"]:
                try:
                    poly_str = construir_polinomio(xs, ys, nombre)
                    polinomios[nombre] = poly_str
                except:
                    polinomios[nombre] = "Error al construir polinomio"

        except Exception as e:
            print(f"Error en método {nombre}: {str(e)}")
            resultados.append({"metodo": nombre, "resultado": float('nan'), "error": float('inf')})

    resultados_ordenados = sorted(resultados, key=lambda r: r['error'])
    mejor_metodo = resultados_ordenados[0]['metodo'] if resultados_ordenados else "Ninguno"
    return resultados, mejor_metodo, polinomios
