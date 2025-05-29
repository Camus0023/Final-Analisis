from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
import math
import time
from datetime import datetime
import json

# Importar métodos de cada capítulo
from capitulo1 import NumericalMethods, create_graph as create_graph_cap1, determine_best_method
from capitulo2 import (jacobi_method, gauss_seidel_method, sor_method, 
                      format_result, generar_informe as generar_informe_cap2)
from capitulo3 import (vandermonde, newton, lagrange, spline_lineal, spline_cubico,
                      construir_polinomio, generar_grafico, evaluar_metodos)
from capitulo4 import (trapecio, simpson13, simpson38, parse_function, 
                      create_comparison_report)

app = Flask(__name__)
app.secret_key = "clave_secreta_analisis_numerico"

# Variable global para almacenar resultados
last_results = {}

@app.route('/')
def inicio():
    return render_template('inicio.html')

@app.route('/capitulo1')
def capitulo1():
    return render_template('capitulo1.html')

@app.route('/capitulo2')
def capitulo2():
    return render_template('capitulo2.html')

@app.route('/capitulo3')
def capitulo3():
    return render_template('capitulo3.html')

@app.route('/capitulo4')
def capitulo4():
    return render_template('capitulo4.html')

@app.route('/ayuda')
def ayuda():
    return render_template('ayuda.html')

# Rutas para cálculos de cada capítulo
@app.route('/calculate_cap1', methods=['POST'])
def calculate_cap1():
    global last_results
    
    try:
        # Obtener datos del formulario
        function_str = request.form.get('function', '').strip()
        derivative_str = request.form.get('derivative', '').strip()
        interval_a = float(request.form.get('interval_a', 1))
        interval_b = float(request.form.get('interval_b', 2))
        x0 = float(request.form.get('x0', 1.5))
        tolerance = float(request.form.get('tolerance', 0.0001))
        max_iterations = int(request.form.get('max_iterations', 100))
        methods_list = request.form.getlist('methods')
        
        # Validaciones
        if not function_str:
            return jsonify({'error': 'La función no puede estar vacía'})
        
        if tolerance <= 0:
            return jsonify({'error': 'La tolerancia debe ser mayor que cero'})
        
        if max_iterations <= 0:
            return jsonify({'error': 'El número máximo de iteraciones debe ser mayor que cero'})
        
        # Inicializar métodos
        methods = NumericalMethods(function_str, derivative_str)
        results = {}
        warnings = []
        
        # Ejecutar métodos seleccionados
        for method in methods_list:
            start_time = time.perf_counter()
            
            try:
                if method == 'bisection':
                    root, iterations, error, converged, message = methods.bisection(interval_a, interval_b, tolerance, max_iterations)
                elif method == 'false_position':
                    root, iterations, error, converged, message = methods.false_position(interval_a, interval_b, tolerance, max_iterations)
                elif method == 'fixed_point':
                    root, iterations, error, converged, message = methods.fixed_point(x0, tolerance, max_iterations)
                elif method == 'newton':
                    root, iterations, error, converged, message = methods.newton_raphson(x0, tolerance, max_iterations)
                elif method == 'secant':
                    root, iterations, error, converged, message = methods.secant(interval_a, interval_b, tolerance, max_iterations)
                elif method == 'multiple_roots':
                    root, iterations, error, converged, message = methods.multiple_roots(x0, tolerance, max_iterations)
                else:
                    continue
                
                execution_time = time.perf_counter() - start_time
                
                results[method] = {
                    'root': root,
                    'iterations': iterations,
                    'error': error,
                    'execution_time': execution_time,
                    'converged': converged,
                    'message': message
                }
                
                if not converged and message:
                    warnings.append(f"{method}: {message}")
                    
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                results[method] = {
                    'root': None,
                    'iterations': 0,
                    'error': None,
                    'execution_time': execution_time,
                    'converged': False,
                    'message': f"Error: {str(e)}"
                }
                warnings.append(f"{method}: Error durante la ejecución - {str(e)}")
        
        # Crear gráfica
        graph_base64 = create_graph_cap1(function_str, results, interval_a, interval_b)
        
        # Determinar mejor método
        best_method, reason = determine_best_method(results)
        
        comparison = {
            'best_method': best_method,
            'reason': reason
        } if best_method else None
        
        # Almacenar resultados
        last_results['cap1'] = {
            'function': function_str,
            'derivative': derivative_str,
            'parameters': {
                'interval_a': interval_a,
                'interval_b': interval_b,
                'x0': x0,
                'tolerance': tolerance,
                'max_iterations': max_iterations
            },
            'results': results,
            'comparison': comparison,
            'graph': graph_base64,
            'warnings': warnings,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'results': results,
            'comparison': comparison,
            'graph': graph_base64,
            'warnings': warnings
        })
        
    except Exception as e:
        return jsonify({'error': f'Error general: {str(e)}'})

@app.route('/calculate_cap2', methods=['POST'])
def calculate_cap2():
    global last_results
    
    try:
        A = request.form['A']
        b = request.form['b']
        x0 = request.form['x0']
        tol = float(request.form['tol'])
        niter = int(request.form['niter'])
        w = float(request.form['w'])
        error_type = request.form['error_type']
        metodos = request.form.getlist('metodo')

        if tol <= 0:
            return jsonify({'error': 'La tolerancia debe ser positiva'})
        
        if niter <= 0:
            return jsonify({'error': 'El número de iteraciones debe ser positivo'})

        resultados = {}
        if 'jacobi' in metodos:
            resultados['Jacobi'] = jacobi_method(A, b, x0, tol, niter, error_type)
        if 'gaussseidel' in metodos:
            resultados['Gauss-Seidel'] = gauss_seidel_method(A, b, x0, tol, niter, error_type)
        if 'sor' in metodos:
            resultados['SOR'] = sor_method(A, b, x0, tol, niter, w, error_type)

        resultado_texto = ""
        for nombre, metodo in resultados.items():
            resultado_texto += format_result(nombre, metodo[1], metodo[2], metodo[3], tol=tol) + "\n"

        mejor = None
        if len(resultados) > 1:
            mejor = generar_informe_cap2(resultados)
            resultado_texto += f"\nMejor método: {mejor} (por menor número de iteraciones)"

        last_results['cap2'] = {
            'resultados': resultados,
            'mejor': mejor,
            'texto': resultado_texto,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify({
            'resultado': resultado_texto,
            'informe': len(resultados) > 1
        })

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/calculate_cap3', methods=['POST'])
def calculate_cap3():
    global last_results
    
    try:
        xs, ys = [], []
        for i in range(8):
            x_val = request.form.get(f"x{i}")
            y_val = request.form.get(f"y{i}")
            if x_val and y_val:
                try:
                    xs.append(float(x_val))
                    ys.append(float(y_val))
                except:
                    pass

        if len(xs) < 2:
            return jsonify({'error': 'Se necesitan al menos 2 puntos'})

        # Verificar valores repetidos
        if len(set(xs)) != len(xs):
            return jsonify({'error': 'El vector X no puede tener valores repetidos'})
        
        if len(set(ys)) != len(ys):
            return jsonify({'error': 'El vector Y no puede tener valores repetidos'})

        x_eval = float(request.form.get("x_eval"))
        y_real = request.form.get("y_real")
        y_real = float(y_real) if y_real else None

        metodos_seleccionados = request.form.getlist("metodo")

        todos_los_metodos = {
            "Vandermonde": vandermonde,
            "Newton": newton,
            "Lagrange": lagrange,
            "Spline Lineal": spline_lineal,
            "Spline Cúbico": spline_cubico
        }

        metodos_a_ejecutar = [(nombre, metodo) for nombre, metodo in todos_los_metodos.items() if nombre in metodos_seleccionados]

        resultados, mejor_metodo, polinomios = evaluar_metodos(np.array(xs), np.array(ys), x_eval, y_real, metodos_a_ejecutar)
        grafico = generar_grafico(xs, ys, x_eval, metodos_a_ejecutar)

        # Convertir cualquier objeto ndarray en resultados a lista
        for resultado in resultados:
            for key, value in resultado.items():
                if isinstance(value, np.ndarray):
                    resultado[key] = value.tolist()  # Convertir ndarray a lista

        # Aseguramos que polinomios sean serializables
        for key in polinomios:
            if not isinstance(polinomios[key], str):
                polinomios[key] = str(polinomios[key])

        last_results['cap3'] = {
            'resultados': resultados,
            'mejor_metodo': mejor_metodo,
            'polinomios': polinomios,
            'grafico': grafico,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify({
            'resultados': resultados,
            'mejor_metodo': mejor_metodo,
            'polinomios': polinomios,
            'grafico': grafico
        })

    except Exception as e:
        print(f"Error en calculate_cap3: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/calculate_cap4', methods=['POST'])
def calculate_cap4():
    global last_results
    
    try:
        metodo = request.form.get('metodo')
        func_str = request.form.get('func', '').strip()
        a = float(request.form.get('a', '0'))
        b = float(request.form.get('b', '0'))
        n = int(request.form.get('n', '0'))

        if not func_str:
            return jsonify({'error': 'La función no puede estar vacía'})

        f = parse_function(func_str)

        if metodo == 'trapecio':
            if n < 1:
                return jsonify({'error': 'n debe ser ≥ 1 para el método del trapecio'})
            resultado = trapecio(f, a, b, n)
        elif metodo == 'simpson13':
            if n < 2 or n % 2 != 0:
                return jsonify({'error': 'n debe ser par y ≥ 2 para Simpson 1/3'})
            resultado = simpson13(f, a, b, n)
        elif metodo == 'simpson38':
            if n < 3 or n % 3 != 0:
                return jsonify({'error': 'n debe ser múltiplo de 3 y ≥ 3 para Simpson 3/8'})
            resultado = simpson38(f, a, b, n)
        else:
            return jsonify({'error': 'Método no válido'})

        # Crear gráfica
        h = (b - a) / n
        x_points = [a + i * h for i in range(n + 1)]
        y_points = [f(xi) for xi in x_points]
        x_curve = [a + i * (b - a) / 200 for i in range(201)]
        y_curve = [f(xi) for xi in x_curve]

        plt.figure(figsize=(10, 6))
        plt.plot(x_curve, y_curve, 'b-', linewidth=2, label=f'f(x) = {func_str}')
        plt.plot(x_points, y_points, 'ro-', markersize=6, label=f'{metodo.title()}')
        plt.fill_between(x_points, 0, y_points, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Método: {metodo.title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        last_results['cap4'] = {
            'metodo': metodo,
            'funcion': func_str,
            'a': a,
            'b': b,
            'n': n,
            'resultado': resultado,
            'grafico': img_base64,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify({
            'resultado': resultado,
            'grafico': img_base64
        })

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/comparar_cap4', methods=['POST'])
def comparar_cap4():
    try:
        func_str = request.form.get('func', '').strip()
        a = float(request.form.get('a', '0'))
        b = float(request.form.get('b', '0'))
        n_trap = int(request.form.get('n_trap', '0'))
        n_s13 = int(request.form.get('n_s13', '0'))
        n_s38 = int(request.form.get('n_s38', '0'))

        if not func_str:
            return jsonify({'error': 'La función no puede estar vacía'})

        f = parse_function(func_str)

        # Validaciones
        if n_trap < 1:
            return jsonify({'error': 'n para trapecio debe ser ≥ 1'})
        if n_s13 < 2 or n_s13 % 2 != 0:
            return jsonify({'error': 'n para Simpson 1/3 debe ser par y ≥ 2'})
        if n_s38 < 3 or n_s38 % 3 != 0:
            return jsonify({'error': 'n para Simpson 3/8 debe ser múltiplo de 3 y ≥ 3'})

        # Calcular resultados
        resultado_trap = trapecio(f, a, b, n_trap)
        resultado_s13 = simpson13(f, a, b, n_s13)
        resultado_s38 = simpson38(f, a, b, n_s38)

        # Crear reporte PDF
        pdf_buffer = create_comparison_report(func_str, a, b, n_trap, n_s13, n_s38, 
                                            resultado_trap, resultado_s13, resultado_s38, f)

        return send_file(pdf_buffer, as_attachment=True, 
                        download_name=f'reporte_integracion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                        mimetype='application/pdf')

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('capitulo4'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
