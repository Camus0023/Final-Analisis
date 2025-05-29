import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import math
import time

class NumericalMethods:
    def __init__(self, function_str, derivative_str=None):
        self.function_str = function_str
        self.derivative_str = derivative_str
        
    def safe_eval(self, expr, x_val):
        """Safely evaluate mathematical expressions"""
        try:
            if expr is None or expr.strip() == '':
                return None
            
            # Replace common mathematical functions and operators
            expr = expr.replace('^', '**')
            expr = expr.replace('ln', 'log')
            
            # Create a safe namespace
            namespace = {
                'x': float(x_val),
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'exp': math.exp,
                'log': math.log,
                'sqrt': math.sqrt,
                'abs': abs,
                'pi': math.pi,
                'e': math.e,
                'pow': pow
            }
            
            result = eval(expr, {"__builtins__": {}}, namespace)
            
            # Check for invalid results
            if math.isnan(result) or math.isinf(result):
                return None
                
            return float(result)
        except Exception as e:
            print(f"Error evaluating expression '{expr}' at x={x_val}: {e}")
            return None
    
    def f(self, x):
        return self.safe_eval(self.function_str, x)
    
    def df(self, x):
        if self.derivative_str:
            return self.safe_eval(self.derivative_str, x)
        return None
    
    def bisection(self, a, b, tolerance=1e-6, max_iterations=100):
        """Método de Bisección"""
        try:
            fa = self.f(a)
            fb = self.f(b)
            
            if fa is None or fb is None:
                return None, 0, None, False, "Error evaluando la función en los extremos del intervalo"
            
            if fa * fb >= 0:
                return None, 0, None, False, "La función no cambia de signo en el intervalo dado"
            
            iterations = 0
            c = a
            
            while iterations < max_iterations:
                c = (a + b) / 2
                fc = self.f(c)
                
                if fc is None:
                    return None, iterations, None, False, "Error evaluando la función"
                
                error = abs(b - a) / 2
                iterations += 1
                
                if error < tolerance or abs(fc) < tolerance:
                    return c, iterations, error, True, "Convergió exitosamente"
                
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            
            return c, iterations, abs(b - a) / 2, False, "Máximo número de iteraciones alcanzado"
            
        except Exception as e:
            return None, 0, None, False, f"Error en bisección: {str(e)}"
    
    def false_position(self, a, b, tolerance=1e-6, max_iterations=100):
        """Método de Regla Falsa"""
        try:
            fa = self.f(a)
            fb = self.f(b)
            
            if fa is None or fb is None:
                return None, 0, None, False, "Error evaluando la función en los extremos del intervalo"
            
            if fa * fb >= 0:
                return None, 0, None, False, "La función no cambia de signo en el intervalo dado"
            
            iterations = 0
            c = a
            c_prev = a
            
            while iterations < max_iterations:
                # Calculate new approximation
                c = b - fb * (b - a) / (fb - fa)
                fc = self.f(c)
                
                if fc is None:
                    return None, iterations, None, False, "Error evaluando la función"
                
                iterations += 1
                
                # Calculate error
                if iterations > 1:
                    error = abs(c - c_prev)
                else:
                    error = abs(b - a)
                
                if error < tolerance or abs(fc) < tolerance:
                    return c, iterations, error, True, "Convergió exitosamente"
                
                # Update interval
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
                
                c_prev = c
            
            return c, iterations, error, False, "Máximo número de iteraciones alcanzado"
            
        except Exception as e:
            return None, 0, None, False, f"Error en regla falsa: {str(e)}"
    
    def fixed_point(self, x0, tolerance=1e-6, max_iterations=100):
        """Método de Punto Fijo"""
        try:
            k = 2.0
            iterations = 0
            x = x0
            
            while iterations < max_iterations:
                fx = self.f(x)
                
                if fx is None:
                    return None, iterations, None, False, "Error evaluando la función"
                
                x_new = x - fx / k
                
                iterations += 1
                error = abs(x_new - x)
                
                if error < tolerance or abs(fx) < tolerance:
                    return x_new, iterations, error, True, "Convergió exitosamente"
                
                if abs(x_new) > 1e10:
                    return None, iterations, error, False, "El método diverge"
                
                x = x_new
            
            return x, iterations, error, False, "Máximo número de iteraciones alcanzado"
            
        except Exception as e:
            return None, 0, None, False, f"Error en punto fijo: {str(e)}"
    
    def newton_raphson(self, x0, tolerance=1e-6, max_iterations=100):
        """Método de Newton-Raphson"""
        try:
            if not self.derivative_str or self.derivative_str.strip() == '':
                return None, 0, None, False, "Se requiere la derivada para el método de Newton"
            
            iterations = 0
            x = x0
            
            while iterations < max_iterations:
                fx = self.f(x)
                dfx = self.df(x)
                
                if fx is None or dfx is None:
                    return None, iterations, None, False, "Error evaluando la función o su derivada"
                
                if abs(dfx) < 1e-12:
                    return None, iterations, None, False, "Derivada muy pequeña (posible punto crítico)"
                
                x_new = x - fx / dfx
                iterations += 1
                error = abs(x_new - x)
                
                if error < tolerance or abs(fx) < tolerance:
                    return x_new, iterations, error, True, "Convergió exitosamente"
                
                if abs(x_new) > 1e10:
                    return None, iterations, error, False, "El método diverge"
                
                x = x_new
            
            return x, iterations, error, False, "Máximo número de iteraciones alcanzado"
            
        except Exception as e:
            return None, 0, None, False, f"Error en Newton-Raphson: {str(e)}"
    
    def secant(self, x0, x1, tolerance=1e-6, max_iterations=100):
        """Método de la Secante"""
        try:
            iterations = 0
            
            while iterations < max_iterations:
                fx0 = self.f(x0)
                fx1 = self.f(x1)
                
                if fx0 is None or fx1 is None:
                    return None, iterations, None, False, "Error evaluando la función"
                
                if abs(fx1 - fx0) < 1e-12:
                    return None, iterations, None, False, "División por cero (puntos muy cercanos)"
                
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                
                iterations += 1
                error = abs(x2 - x1)
                
                if error < tolerance or abs(fx1) < tolerance:
                    return x2, iterations, error, True, "Convergió exitosamente"
                
                if abs(x2) > 1e10:
                    return None, iterations, error, False, "El método diverge"
                
                x0, x1 = x1, x2
            
            return x1, iterations, error, False, "Máximo número de iteraciones alcanzado"
            
        except Exception as e:
            return None, 0, None, False, f"Error en secante: {str(e)}"
    
    def multiple_roots(self, x0, tolerance=1e-6, max_iterations=100):
        """Método para Raíces Múltiples"""
        try:
            if not self.derivative_str or self.derivative_str.strip() == '':
                return None, 0, None, False, "Se requiere la derivada para el método de raíces múltiples"
            
            iterations = 0
            x = x0
            h = 1e-8
            
            while iterations < max_iterations:
                fx = self.f(x)
                dfx = self.df(x)
                
                if fx is None or dfx is None:
                    return None, iterations, None, False, "Error evaluando la función o su derivada"
                
                if abs(dfx) < 1e-12:
                    return None, iterations, None, False, "Derivada muy pequeña"
                
                dfx_plus = self.df(x + h)
                dfx_minus = self.df(x - h)
                
                if dfx_plus is None or dfx_minus is None:
                    return None, iterations, None, False, "Error calculando la segunda derivada"
                
                d2fx = (dfx_plus - dfx_minus) / (2 * h)
                
                denominator = dfx**2 - fx * d2fx
                if abs(denominator) < 1e-12:
                    return None, iterations, None, False, "División por cero en el método modificado"
                
                x_new = x - (fx * dfx) / denominator
                
                iterations += 1
                error = abs(x_new - x)
                
                if error < tolerance or abs(fx) < tolerance:
                    return x_new, iterations, error, True, "Convergió exitosamente"
                
                if abs(x_new) > 1e10:
                    return None, iterations, error, False, "El método diverge"
                
                x = x_new
            
            return x, iterations, error, False, "Máximo número de iteraciones alcanzado"
            
        except Exception as e:
            return None, 0, None, False, f"Error en raíces múltiples: {str(e)}"

def create_graph(function_str, results, interval_a, interval_b):
    """Create a graph of the function and roots"""
    try:
        plt.figure(figsize=(12, 8))
        
        x_min = min(interval_a, interval_b) - 2
        x_max = max(interval_a, interval_b) + 2
        x = np.linspace(x_min, x_max, 1000)
        
        methods = NumericalMethods(function_str)
        y = []
        
        for xi in x:
            yi = methods.f(xi)
            if yi is not None and not (math.isnan(yi) or math.isinf(yi)):
                y.append(yi)
            else:
                y.append(np.nan)
        
        y = np.array(y)
        
        plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {function_str}')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        method_names = {
            'bisection': 'Bisección',
            'false_position': 'Regla Falsa',
            'fixed_point': 'Punto Fijo',
            'newton': 'Newton-Raphson',
            'secant': 'Secante',
            'multiple_roots': 'Raíces Múltiples'
        }
        
        plotted_roots = []
        for i, (method, result) in enumerate(results.items()):
            if result['root'] is not None and result['converged']:
                root = result['root']
                is_duplicate = any(abs(root - pr) < 0.001 for pr in plotted_roots)
                if not is_duplicate:
                    color = colors[i % len(colors)]
                    plt.plot(root, 0, 'o', color=color, markersize=10, 
                            label=f'{method_names.get(method, method)}: {root:.4f}')
                    plotted_roots.append(root)
        
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title('Función y Raíces Encontradas', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    except Exception as e:
        print(f"Error creating graph: {e}")
        return None

def determine_best_method(results):
    """Determine the most effective method"""
    converged_methods = {k: v for k, v in results.items() if v['converged'] and v['root'] is not None}
    
    if not converged_methods:
        return None, "Ningún método convergió"
    
    best_method = min(converged_methods.keys(), 
                     key=lambda k: (
                         converged_methods[k]['iterations'], 
                         converged_methods[k]['execution_time'],
                         converged_methods[k]['error'] if converged_methods[k]['error'] is not None else float('inf')
                     ))
    
    result = converged_methods[best_method]
    reason = f"Convergió en {result['iterations']} iteraciones en {result['execution_time']*1000:.3f} ms"
    
    return best_method, reason
