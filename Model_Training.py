import Cleaner
import plotly.express as px
import warnings
import pandas as pd
from EmpiricDistribution import EmpiricDistribution
import pickle # Para guardar los modelos entrenados
from datetime import time 
import matplotlib.pyplot as plt
import numpy as np
import random
from sympy import Symbol
from datetime import time

class Model_Training:
    def __init__(self):
        pass

    def generate_models(self):
        #Importing data
        cleaner = Cleaner.Cleaner()
        cleaner.set_path_info({'7': '07_v2.xlsx', '50':'50.xlsx'})
        df = cleaner.process_data(export_csv=True)

        #Small check
        df['hora_salida'] = pd.to_datetime(df['hora_salida'], format='%H:%M').dt.time
        df = df.dropna(subset=['Usuarios']) # Limpieza rápida si hay NaNs

        # Definición de Segmentos
        TRAYECTOS = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2']
        TIME_INTERVALS = [
            ('07:00', '09:30'), 
            ('13:30', '18:00'), 
            ('06:00', '07:00'), 
            ('09:30', '13:30'), 
            ('18:30', '22:15')
        ]

        # CANTIDAD DE PASAJEROS

        # Diccionario para almacenar todos los modelos entrenados
        pax_models = {}

        # Modelado Iterativo para cada segmento (Ruta + Hora)
        for trayecto in TRAYECTOS:
            sub_trayecto = df.loc[df['trayecto'] == trayecto]
            
            for time_start, time_end in TIME_INTERVALS:
                # Filtrar datos por el intervalo de tiempo
                sub_time = sub_trayecto.loc[
                    (sub_trayecto['hora_salida'].astype(str) >= time_start) & 
                    (sub_trayecto['hora_salida'].astype(str) < time_end)
                ]
                
                # Clave única para guardar el modelo
                model_key = f"{trayecto}_{time_start.replace(':', '')}-{time_end.replace(':', '')}"

                if not sub_time.empty and sub_time['Usuarios'].max() > 0:
                    pax_data = sub_time['Usuarios'].tolist()
                    
                    # --- Visualización (Opcional, pero recomendado) ---
                    # Muestra la distribución de pasajeros para este segmento
                    max_pax = int(sub_time['Usuarios'].max())
                    fig = px.histogram(
                        sub_time, 
                        x='Usuarios', 
                        nbins=max_pax + 1,
                        title=f'Distribución de Pax: {trayecto} ({time_start}-{time_end})',
                        labels={'Usuarios': 'Pasajeros'}
                    )
                    print(f"Generando histograma para {model_key}")
                    # 
                    # fig.show() 
                    
                    # 4. Entrenar el Modelo Empírico
                    pax_model = EmpiricDistribution()
                    
                    # Para variables discretas (pasajeros), n_bins debe ser max_pax + 1 
                    # para que cada número entero (0, 1, 2, ...) sea un bin.
                    pax_model.set_histogram(pax_data, n_bins=max_pax + 1)
                    pax_model.train(show_process=False) # Entrenar la transformada inversa
                    
                    # Almacenar el modelo entrenado
                    pax_models[model_key] = pax_model
                    print(f"Modelo {model_key} entrenado. Número de segmentos (inversas): {len(pax_model.inverses)}")
                else:
                    # Manejar el caso donde no hay datos o todos los Usuarios son 0
                    pax_models[model_key] = None
                    print(f"Advertencia: No hay datos suficientes o Usuarios son cero para {model_key}")


        # Guardar el diccionario de modelos para usarlo en el script de simulación principal.
        with open('pax_models.pkl', 'wb') as file:
            pickle.dump(pax_models, file)

        print("\nProceso finalizado. Los modelos de pasajeros han sido guardados en 'pax_models.pkl'")


        #TIEMPOS DE VIAJE DE AUTOBUSES

        time_models = {}

        # Modelado Iterativo
        print("Iniciando modelado de la distribución de Tiempos de Viaje...")

        for trayecto in TRAYECTOS:
            sub_trayecto = df.loc[df['trayecto'] == trayecto]
            
            for time_start_str, time_end_str in TIME_INTERVALS:
                
                # Convertir strings de hora a objetos time para la comparación
                time_start = time.fromisoformat(time_start_str)
                time_end = time.fromisoformat(time_end_str)
                
                # Filtrar datos por el intervalo de tiempo
                sub_time = sub_trayecto.loc[
                    (sub_trayecto['hora_salida'] >= time_start) & 
                    (sub_trayecto['hora_salida'] < time_end)
                ]
                
                model_key = f"{trayecto}_{time_start_str.replace(':', '')}-{time_end_str.replace(':', '')}"

                if not sub_time.empty and sub_time['minutos_viaje'].count() > 5:
                    time_data = sub_time['minutos_viaje'].tolist()
                    
                    # Para variables continuas (tiempo), usamos un número fijo de bins (ej. 15)
                    # para modelar la forma de la distribución.
                    time_model = EmpiricDistribution()
                    time_model.set_histogram(time_data, n_bins=15) 
                    
                    try:
                        time_model.train(show_process=False)
                        time_models[model_key] = time_model
                        print(f"  -> Modelo de Tiempo {model_key} entrenado.")
                    except Exception as e:
                        # Captura de errores de Sympy, aunque deberían ser menos comunes aquí
                        print(f"  -> Error al entrenar el modelo de Tiempo {model_key}: {e}")
                        time_models[model_key] = None
                else:
                    time_models[model_key] = None
                    print(f"  -> Advertencia: Sin datos suficientes para modelar {model_key}")


        # Almacenamiento de los Modelos
        with open('time_models.pkl', 'wb') as file:
            pickle.dump(time_models, file)

        print("\n Proceso finalizado. Los modelos de tiempo de viaje han sido guardados en 'time_models.pkl'")

    def generate_simulation(self, iterations = 1000):
        # Definición de Parámetros
        NUM_ITERACIONES = iterations
        TRAYECTOS = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2']
        TIME_INTERVALS = [
            ('07:00', '09:30'), 
            ('13:30', '18:00'), 
            ('06:00', '07:00'), 
            ('09:30', '13:30'), 
            ('18:30', '22:15') # Este tramo será sustituido con lógica especial
        ]
        # Clave del segmento usado como proxy para TIEMPO (y fallback general)
        PROXY_KEY = 'A2-GLAXO_0600-0700' 

        # Carga de Modelos
        try:
            with open('time_models.pkl', 'rb') as file:
                time_models = pickle.load(file)
            with open('pax_models.pkl', 'rb') as file:
                pax_models = pickle.load(file)
            print("Modelos cargados correctamente.")
            
            # Definir los Modelos Sustitutos Globales (Base)
            MODELO_TIEMPO_SUSTITUTO = time_models.get(PROXY_KEY)
            MODELO_PAX_SUSTITUTO = pax_models.get(PROXY_KEY)

            if not MODELO_TIEMPO_SUSTITUTO or not MODELO_PAX_SUSTITUTO:
                print(f"Error: El modelo proxy ({PROXY_KEY}) no se pudo cargar. Verifique que exista en ambos .pkl.")
                exit()

        except FileNotFoundError:
            print("Error: Asegúrate de que los archivos .pkl existan en el directorio.")
            exit()

        # Ejecución de la Simulación
        simulated_data = []

        print(f"\nIniciando simulación de Monte Carlo con {NUM_ITERACIONES} iteraciones por segmento...")

        for trayecto in TRAYECTOS:
            for time_start, time_end in TIME_INTERVALS:
                
                # Generar la clave única actual
                model_key = f"{trayecto}_{time_start.replace(':', '')}-{time_end.replace(':', '')}"
                
                time_model = time_models.get(model_key)
                pax_model = pax_models.get(model_key)
                
                # Factor de escala por defecto (100%)
                pax_scale_factor = 1.0 

                
                if not time_model or not pax_model or '1830-2215' in model_key:
                    
                    # Condición específica para el tramo nocturno (18:30 - 22:15)
                    if '1830-2215' in model_key:
                        
                        # 1. TIEMPO: Usamos el proxy original (06:00-07:00) tal como pediste
                        time_model = MODELO_TIEMPO_SUSTITUTO
                        
                        # 2. PAX: Usamos el tramo 13:30-18:00 del MISMO trayecto como referencia
                        # Construimos la key de referencia
                        ref_pax_key = f"{trayecto}_1330-1800"
                        ref_pax_model = pax_models.get(ref_pax_key)
                        
                        if ref_pax_model:
                            pax_model = ref_pax_model
                            pax_scale_factor = 0.80 # 20% más tranquilo = 80% del volumen
                            print(f"  -> Sustitución PAX en {model_key}: Usando base {ref_pax_key} al 80%.")
                            print(f"  -> Sustitución TIEMPO en {model_key}: Usando proxy {PROXY_KEY}.")
                        else:
                            # Fallback por si el modelo de la tarde tampoco existe
                            pax_model = MODELO_PAX_SUSTITUTO
                            print(f"  -> Advertencia: Modelo referencia {ref_pax_key} no hallado. Usando proxy global.")

                    else:
                        # Si es otro segmento diferente que falta y no es el nocturno
                        if not time_model or not pax_model:
                            print(f"  -> Advertencia: Saltando segmento {model_key} por falta de modelo.")
                            continue 
                
                # Generar muestras usando el modelo
                try:
                    # Generamos tiempos
                    simulated_times = time_model.generate_samples(NUM_ITERACIONES)
                    
                    # Generamos pasajeros y APLICAMOS EL FACTOR DE ESCALA
                    # Si el factor es 1.0, queda igual. Si es 0.8, reduce un 20%.
                    raw_pax_samples = pax_model.generate_samples(NUM_ITERACIONES)
                    simulated_pax = [round(pax * pax_scale_factor) for pax in raw_pax_samples]
                    
                except Exception as e:
                    print(f"  -> Error durante el muestreo del segmento {model_key}: {e}")
                    continue

                min_len = min(len(simulated_times), len(simulated_pax))
                    
                # Almacenar los resultados simulados
                df_segment = pd.DataFrame({
                    'trayecto': model_key,
                    'minutos_viaje_simulado': simulated_times[:min_len],
                    'usuarios_simulado': simulated_pax[:min_len],
                    'iteracion': range(min_len)
                })
                simulated_data.append(df_segment)
                    
                # Solo imprimimos confirmación si no se imprimió ya en la lógica de sustitución
                if '1830-2215' not in model_key:
                    print(f"  -> Segmento {model_key} simulado con {min_len} muestras.")

                
        # Consolidación y Exportación
        if simulated_data:
            df_simulacion_final = pd.concat(simulated_data, ignore_index=True)
            output_file = 'simulation_results_completa.csv'
            df_simulacion_final.to_csv(output_file, index=False)
            
            print("\n=======================================================")
            print(f"Simulación de Monte Carlo COMPLETADA.")
            print(f"Total de registros simulados: {len(df_simulacion_final)}")
            print(f"Resultados guardados en: {output_file}")
            print("=======================================================")
        else:
            print("\nSimulación finalizada sin generar datos. Revise los archivos .pkl.")
    

    
    def show_results(self):
            # Carga y Pre-procesamiento Inicial de Datos
        try:
            # Intenta cargar el archivo de simulación más reciente
            df_simulacion = pd.read_csv('simulation_results_completa.csv')
        except FileNotFoundError:
            try:
                # Fallback al archivo anterior si el completo no existe
                df_simulacion = pd.read_csv('simulation_results.csv')
            except FileNotFoundError:
                print("Error: No se encontró 'simulation_results_completa.csv' ni 'simulation_results.csv'.")
                df_simulacion = None

        try:
            df_original = pd.read_csv('data_clean.csv')
            # Pre-procesar la columna hora_salida en df_original una vez
            df_original['hora_salida'] = pd.to_datetime(df_original['hora_salida'], format='%H:%M', errors='coerce').dt.time
        except FileNotFoundError:
            print("Error: No se encontró 'data_clean.csv'.")
            df_original = None
        except Exception as e:
            print(f"Error durante el pre-procesamiento inicial de df_original: {e}")
            df_original = None

        # Función de Visualización

        def graficar_histograma_pax_comparativo(
            segmento_clave: str, 
            df_simulacion: pd.DataFrame, 
            df_original: pd.DataFrame
        ):
            """
            Genera un histograma comparativo del número de pasajeros entre
            los datos históricos originales y los datos simulados por Monte Carlo.
            """
            if df_simulacion is None or df_original is None:
                print("No se pudo cargar uno o ambos DataFrames. Saltando gráfica.")
                return

            # Preparación de Claves
            parts = segmento_clave.split('_')
            if len(parts) != 2:
                return # Saltar si la clave es inválida
                
            trayecto_original = parts[0]
            time_str = parts[1]
            
            try:
                time_start_str, time_end_str = time_str.split('-')
            except ValueError:
                return

            # Convertir 'HHMM' a 'HH:MM' y a objeto time
            def format_time_str(s):
                return f"{s[:2]}:{s[2:]}"

            try:
                time_start = time.fromisoformat(format_time_str(time_start_str))
                time_end = time.fromisoformat(format_time_str(time_end_str))
            except ValueError:
                return

            # Filtrado de Datos
            
            pax_simulado = df_simulacion[df_simulacion['trayecto'] == segmento_clave]['usuarios_simulado']

            # Se usa .loc y la columna 'hora_salida' pre-procesada
            pax_original_trayecto = df_original[df_original['trayecto'] == trayecto_original].copy()
            
            pax_original_segmento = pax_original_trayecto.loc[
                (pax_original_trayecto['hora_salida'] >= time_start) & 
                (pax_original_trayecto['hora_salida'] < time_end)
            ]['Usuarios'].dropna()

            if pax_simulado.empty:
                print(f"Advertencia: No hay datos simulados para {segmento_clave}. Saltando gráfica.")
                return

            # Generación del Histograma
            
            plt.figure(figsize=(10, 6))

            # Calcular el número máximo de pasajeros para establecer los bins
            max_pax_sim = pax_simulado.max()
            max_pax_orig = pax_original_segmento.max() if not pax_original_segmento.empty else 0
            max_pax = int(max(max_pax_sim, max_pax_orig))
            
            bins = np.arange(-0.5, max_pax + 1.5, 1) 

            # Histograma de datos originales (solo si hay datos)
            if not pax_original_segmento.empty:
                plt.hist(
                    pax_original_segmento, 
                    bins=bins, 
                    density=True, 
                    alpha=0.6, 
                    label='Datos Originales (Histórico)', 
                    color='skyblue', 
                    edgecolor='black'
                )

            # Histograma de datos simulados
            plt.hist(
                pax_simulado, 
                bins=bins, 
                density=True, 
                alpha=0.4, 
                label='Datos Simulados (Monte Carlo)', 
                color='red', 
                edgecolor='black'
            )

            # Etiquetas y Títulos
            plt.title(f'Distribución de Pasajeros: {segmento_clave}', fontsize=16)
            plt.xlabel('Número de Pasajeros (Usuarios)', fontsize=12)
            plt.ylabel('Densidad de Probabilidad', fontsize=12)
            plt.xticks(np.arange(0, max_pax + 1, 2))
            plt.legend()
            plt.grid(axis='y', alpha=0.5)

            plt.show()
        
        # Bucle para Imprimir todas las Gráficas
        if df_simulacion is not None and df_original is not None:
            todos_los_segmentos = df_simulacion['trayecto'].unique()

            print(f"Iniciando generación de {len(todos_los_segmentos)} histogramas...")
            
            for segmento in todos_los_segmentos:
                # Llama a la función para cada segmento
                graficar_histograma_pax_comparativo(segmento, df_simulacion, df_original)