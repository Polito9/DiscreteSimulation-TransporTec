import numpy as np
from sympy import *
from IPython.display import display, Math
import plotly.graph_objects as go
import random

class EmpiricDistribution:
    start = 0
    end = 0
    bins = 0
    heights = []
    ranges_probs = []
    inverses = []

    def __init__(self):
        pass
    
    def set_histogram(self, data, n_bins=15):
        #Setting histogram information
        counts, bin_edges = np.histogram(data, bins=n_bins)
        self.start = bin_edges[0]
        self.end = bin_edges[-1]
        self.bins = len(counts)

        self.heights = counts

        #print(self.heights)
        #print(bin_edges)
        #print(self.bins)

    def train(self, show_process = False, show_histogram = False):
        #Initial data
        start = self.start
        end = self.end
        bins = self.bins
        heights = self.heights

        bin_width = (end - start) / bins
        points = []

        #First point
        points.append((float(start), 0.0))

        #Middle points
        for i in range(bins):
            center_x = start + bin_width * (i + 0.5)
            points.append((center_x, heights[i]))

        #Last point
        points.append((float(end), 0.0))

        # Area using Shoelace Formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        final_area = abs(area) / 2.0

        if(show_process):
            print(f'Area calculada: {final_area}')

        #Generating equations
        y = Symbol('y')
        x = Symbol('x')

        f = []          # Original functions
        f_scaled = []   # Normalized funtions (PDF)
        intervals = []  # Intervals

        # Creating rects
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            # m = (y2 - y1) / (x2 - x1)
            m_val = (p2[1] - p1[1]) / (p2[0] - p1[0])
            
            # b = y - mx 
            b_val = p1[1] - m_val * p1[0]
            
            # original: y = mx + b
            eq = m_val * x + b_val
            f.append(eq)
            
            # Scaled y = (m/Area)x + (b/Area)
            eq_scaled = (m_val / final_area) * x + (b_val / final_area)
            f_scaled.append(eq_scaled)
            
            intervals.append((p1[0], p2[0]))

        
        if(show_process):
            print('----------------------------------------- \nEcuaciones (Sin escalar): \n ----------------------------------------')
            for i, (eq, interval) in enumerate(zip(f, intervals)):
                display(Math(rf"({i+1}): \; {latex(N(eq, 4))}, \; {interval[0]:.2f} \leq x \leq {interval[1]:.2f}"))

            print('----------------------------------------- \nEcuaciones (Escaladas / PDF): \n ----------------------------------------')
            for i, (eq, interval) in enumerate(zip(f_scaled, intervals)):
                display(Math(rf"({i+1}): \; {latex(N(eq, 4))}, \; {interval[0]:.2f} \leq x \leq {interval[1]:.2f}"))

        if(show_histogram):
            x_vals, y_vals = [], []

            # Muestreo de la función Piecewise para el gráfico
            for expr, (a, b) in zip(f, intervals):
                func = lambdify(x, expr, 'numpy')
                xs = np.linspace(a, b, 50)
                ys = func(xs)
                
                # Truco: si la función es constante (pendiente 0), lambdify devuelve un escalar, no un array
                if np.isscalar(ys): 
                    ys = np.full_like(xs, ys)
                    
                x_vals.extend(xs)
                y_vals.extend(ys)

            fig = go.Figure()

            bin_edges = np.linspace(start, end, bins+1)

            # Histograma de fondo
            fig.add_trace(go.Bar(
                x=(bin_edges[:-1] + bin_edges[1:]) / 2,
                y=heights,
                width=bin_width, # Usamos el width calculado en float
                opacity=0.4,
                name="Histograma",
                marker_color='blue'
            ))

            # Función lineal a trozos
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Función Lineal a Trozos",
                line=dict(color="red", width=2)
            ))

            fig.update_layout(
                title="Histograma con Interpolación Lineal (Floats)",
                xaxis_title="x",
                yaxis_title="y",
                bargap=0,
                template="plotly_white"
            )

            fig.show()

        x, t, y = symbols('x t y')
        
        #Calclating integrals
        integrals = []
        self.ranges_probs = []

        if(show_process): 
            print('----------------------------------------- \n Distribución acumulada F(t): \n ----------------------------------------')

        current_cumulative_prob = 0.0

        for i in range(len(f_scaled)):
            actual_func = f_scaled[i]
            ac_range = intervals[i] # ac_range is (x_start, x_end)
            
            # Integrate the PDF from the start of the bin (ac_range[0]) to t
            # This gives us the area under the curve only within this bin accumulated up to t
            segment_integral = integrate(actual_func, (x, ac_range[0], t))
            
            # The CDF at this point is: (Accumulated from previous bins) + (Current integral)
            cdf_eq = current_cumulative_prob + segment_integral
            
            integrals.append(cdf_eq)
            
            # new cumulative value for the next cycle by evaluating t at the end of the bin
            next_cumulative_prob = float(cdf_eq.subs(t, ac_range[1]))
            
            # Store the probability range covered by this equation [Prob_start, Prob_end]
            self.ranges_probs.append([current_cumulative_prob, next_cumulative_prob])
            
            if(show_process):
                display(Math(rf"({i+1}) \; F(t) = {latex(N(cdf_eq, 4))}, \quad {ac_range[0]:.2f} \leq t \leq {ac_range[1]:.2f}"))
            
            # Update the cumulative total
            current_cumulative_prob = next_cumulative_prob


        # F^-1(y)

        self.inverses = []
        if(show_process):
            print("\n----------------------------------------- \n Inversa F^{-1}(y): \n ----------------------------------------")

        for i in range(len(integrals)):
            # We solve F(t) = y for t. This gives us t as a function of y.
            # Since F(t) is quadratic (linear integral), there are usually two solutions.
            solutions = solve(integrals[i] - y, t)
            
            # Datos para validación
            y_min, y_max = self.ranges_probs[i]
            x_min, x_max = intervals[i]
            
            valid_sol = None
            
            # We select a safe test point (the center of the probability interval).
            test_y = (y_min + y_max) / 2.0
            
            # We verify which solution falls within the correct x range.
            for sol in solutions:
                try:
                    # We evaluate the candidate solution with the test case.
                    val_x = float(sol.subs(y, test_y))
                    
                    # We use a small margin of error (epsilon) for floating point issues.
                    epsilon = 1e-5
                    if (x_min - epsilon) <= val_x <= (x_max + epsilon):
                        valid_sol = sol
                        break
                except TypeError:
                    # If the solution is complex or the conversion fails
                    continue
                    
            if valid_sol is not None:
                self.inverses.append(valid_sol)
                if(show_process):
                    display(Math(rf"({i+1}) \; t = {latex(N(valid_sol, 4))}, \quad {y_min:.4f} \leq y \leq {y_max:.4f}"))
            else:
                print(f"Error: No valid solution for the interval {i+1}")
                self.inverses.append(solutions[0])

    def generate_samples(self, n_samples):
        samples = []
        y = Symbol('y')

        for _ in range(n_samples):
            randi = random.uniform(0, 1)
            print(randi)
            for i in range(len(self.inverses)):
                if(randi>=self.ranges_probs[i][0] and randi <= self.ranges_probs[i][1]):
                    val = float(self.inverses[i].subs(y, randi))
                    samples.append(val)

        return samples