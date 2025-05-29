from sympy import sympify, symbols, integrate, N, E, pi, lambdify
import math
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.utils import ImageReader

def trapecio(f, a, b, n):
    if n < 1: 
        raise ValueError("n debe ser ≥ 1")
    sign = 1
    if b < a: 
        a, b, sign = b, a, -1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += 2 * f(a + i * h)
    return sign * total * h / 2

def simpson13(f, a, b, n):
    if n < 2 or n % 2 != 0: 
        raise ValueError("n debe ser par y ≥ 2")
    sign = 1
    if b < a: 
        a, b, sign = b, a, -1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return sign * total * h / 3

def simpson38(f, a, b, n):
    if n < 3 or n % 3 != 0: 
        raise ValueError("n debe ser múltiplo de 3 y ≥ 3")
    sign = 1
    if b < a: 
        a, b, sign = b, a, -1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (2 if i % 3 == 0 else 3) * f(a + i * h)
    return sign * total * 3 * h / 8

x = symbols('x')

def parse_function(expr_str):
    expr_str = expr_str.replace('^', '**')
    expr = sympify(expr_str, locals={'e': E, 'pi': pi})
    f = lambdify(x, expr, modules=['math'])
    f(0.0)  # validate expression
    return f

def create_comparison_report(func_str, a, b, n_trap, n_s13, n_s38, t, s13, s38, f):
    # Calculate exact value
    expr = sympify(func_str, locals={'e': E, 'pi': pi})
    I_exc = integrate(expr, (x, a, b))
    I_real = float(N(I_exc, 12))

    def errs(v):
        ea = abs(I_real - v)
        er = ea / abs(I_real) if I_real else float('inf')
        return ea, er

    ea_t, er_t = errs(t)
    ea_s13, er_s13 = errs(s13)
    ea_s38, er_s38 = errs(s38)

    # Create graph
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [a + i * (b - a) / 200 for i in range(201)]
    ys = [f(xi) for xi in xs]
    
    h1 = (b - a) / n_trap
    trap_x = [a + i * h1 for i in range(n_trap + 1)]
    trap_y = [f(xi) for xi in trap_x]
    
    h2 = (b - a) / n_s13
    s13_x = [a + i * h2 for i in range(n_s13 + 1)]
    s13_y = [f(xi) for xi in s13_x]
    
    h3 = (b - a) / n_s38
    s38_x = [a + i * h3 for i in range(n_s38 + 1)]
    s38_y = [f(xi) for xi in s38_x]

    ax.plot(xs, ys, label="f(x)", linewidth=2)
    ax.plot(trap_x, trap_y, label="Trapecio", marker='o', markersize=4)
    ax.plot(s13_x, s13_y, label="Simpson 1/3", marker='s', markersize=4)
    ax.plot(s38_x, s38_y, label="Simpson 3/8", marker='^', markersize=4)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    buf_img = io.BytesIO()
    fig.savefig(buf_img, format="PNG", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf_img.seek(0)
    img = ImageReader(buf_img)

    # Generate PDF
    buf = io.BytesIO()
    p = pdfcanvas.Canvas(buf, pagesize=letter)
    w, h = letter

    p.setFont("Helvetica-Bold", 16)
    p.drawCentredString(w/2, h-50, "Reporte de Métodos de Integración Numérica")
    p.setFont("Helvetica", 12)
    y = h - 80
    p.drawString(50, y, f"Función: {func_str}")
    y -= 20
    p.drawString(50, y, f"Intervalo: [{a}, {b}]")
    y -= 20
    p.drawString(50, y, f"n: Trapecio={n_trap}, Simpson 1/3={n_s13}, Simpson 3/8={n_s38}")
    y -= 20
    p.drawString(50, y, f"Valor exacto: {I_real:.8f}")
    y -= 30

    p.drawImage(img, 50, y-200, width=500, height=200)
    y -= 220

    # Results table
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Resultados:")
    y -= 20
    p.setFont("Helvetica", 10)
    
    rows = [
        ("Método", "Aproximación", "Error Absoluto", "Error Relativo"),
        ("Trapecio", f"{t:.8f}", f"{ea_t:.2e}", f"{er_t:.2e}"),
        ("Simpson 1/3", f"{s13:.8f}", f"{ea_s13:.2e}", f"{er_s13:.2e}"),
        ("Simpson 3/8", f"{s38:.8f}", f"{ea_s38:.2e}", f"{er_s38:.2e}")
    ]
    
    for row in rows:
        x0 = 50
        for cell in row:
            p.drawString(x0, y, cell)
            x0 += 120
        y -= 15

    # Best method
    best, err = min(
        [("Trapecio", ea_t), ("Simpson 1/3", ea_s13), ("Simpson 3/8", ea_s38)],
        key=lambda t: t[1]
    )
    y -= 20
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, f"Método más preciso: {best} (Error absoluto: {err:.2e})")

    p.showPage()
    p.save()
    buf.seek(0)
    return buf
