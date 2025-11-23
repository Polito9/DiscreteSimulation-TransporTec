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

    def generate_simulation(self, iterations=1000):
        # Definición de Parámetros
        NUM_ITERACIONES = iterations
        TRAYECTOS = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2']
        TIME_INTERVALS = [
            ('07:00', '09:30'), 
            ('13:30', '18:00'), 
            ('06:00', '07:00'), 
            ('09:30', '13:30'), 
            ('18:30', '22:15') # Tramo con lógica especial (sustitución)
        ]
        # Clave del segmento usado como proxy para TIEMPO
        PROXY_KEY = 'A2-GLAXO_0600-0700' 

        # Carga de Modelos
        try:
            with open('time_models.pkl', 'rb') as file:
                time_models = pickle.load(file)
            print("Modelos de tiempo cargados correctamente.")
            
            # Definir el Modelo Sustituto Global (Base)
            MODELO_TIEMPO_SUSTITUTO = time_models.get(PROXY_KEY)

            if not MODELO_TIEMPO_SUSTITUTO:
                print(f"Error: El modelo proxy ({PROXY_KEY}) no se pudo cargar.")
                exit()

        except FileNotFoundError:
            print("Error: Asegúrate de que el archivo 'time_models.pkl' exista en el directorio.")
            exit()

        # Ejecución de la Simulación
        simulated_data = []

        print(f"\nIniciando simulación de Monte Carlo (Solo Tiempos) con {NUM_ITERACIONES} iteraciones...")

        for trayecto in TRAYECTOS:
            for time_start, time_end in TIME_INTERVALS:
                
                # Generar la clave única actual
                model_key = f"{trayecto}_{time_start.replace(':', '')}-{time_end.replace(':', '')}"
                
                time_model = time_models.get(model_key)
                
                # Lógica de selección de modelo
                if not time_model or '1830-2215' in model_key:
                    
                    # Condición específica para el tramo nocturno (18:30 - 22:15)
                    if '1830-2215' in model_key:
                        # Usamos el proxy original (06:00-07:00) para el tiempo
                        time_model = MODELO_TIEMPO_SUSTITUTO
                        print(f"  -> Sustitución TIEMPO en {model_key}: Usando proxy {PROXY_KEY}.")

                    else:
                        # Si es otro segmento diferente que falta
                        print(f"  -> Advertencia: Saltando segmento {model_key} por falta de modelo.")
                        continue 
                
                # Generar muestras usando el modelo
                try:
                    # Generamos tiempos
                    simulated_times = time_model.generate_samples(NUM_ITERACIONES)
                    
                except Exception as e:
                    print(f"  -> Error durante el muestreo del segmento {model_key}: {e}")
                    continue

                # Almacenar los resultados simulados
                df_segment = pd.DataFrame({
                    'trayecto': model_key,
                    'minutos_viaje_simulado': simulated_times,
                    'iteracion': range(len(simulated_times))
                })
                simulated_data.append(df_segment)
                    
                if '1830-2215' not in model_key:
                    print(f"  -> Segmento {model_key} simulado con {len(simulated_times)} muestras.")

                
        # Consolidación y Exportación
        if simulated_data:
            df_simulacion_final = pd.concat(simulated_data, ignore_index=True)
            output_file = 'simulation_results_times.csv'
            df_simulacion_final.to_csv(output_file, index=False)
            
            print("\n=======================================================")
            print(f"Simulación de Tiempos COMPLETADA.")
            print(f"Total de registros simulados: {len(df_simulacion_final)}")
            print(f"Resultados guardados en: {output_file}")
            print("=======================================================")
        else:
            print("\nSimulación finalizada sin generar datos.")
    


    def show_results(self):
        # Carga y Pre-procesamiento Inicial de Datos
        try:
            # Intenta cargar el archivo de simulación de tiempos
            df_simulacion = pd.read_csv('simulation_results_times.csv')
        except FileNotFoundError:
            print("Error: No se encontró 'simulation_results_times.csv'. Ejecuta la simulación primero.")
            df_simulacion = None

        try:
            df_original = pd.read_csv('data_clean.csv')
            # Pre-procesar hora
            df_original['hora_salida'] = pd.to_datetime(df_original['hora_salida'], format='%H:%M', errors='coerce').dt.time
        except FileNotFoundError:
            print("Advertencia: No se encontró 'data_clean.csv'. Se graficará solo la simulación.")
            df_original = None
        except Exception as e:
            print(f"Error al cargar datos originales: {e}")
            df_original = None

        # Función de Visualización (Adaptada para TIEMPOS)

        def graficar_histograma_tiempo_comparativo(
            segmento_clave: str, 
            df_simulacion: pd.DataFrame, 
            df_original: pd.DataFrame
        ):
            """
            Genera un histograma de los TIEMPOS de viaje simulados.
            Si existe df_original y tiene la columna 'minutos_viaje', la compara.
            """
            if df_simulacion is None:
                return

            # Preparación de Claves y Tiempos para filtrado histórico
            parts = segmento_clave.split('_')
            if len(parts) != 2: return
            trayecto_original = parts[0]
            time_str = parts[1]
            try:
                time_start_str, time_end_str = time_str.split('-')
                time_start = time.fromisoformat(f"{time_start_str[:2]}:{time_start_str[2:]}")
                time_end = time.fromisoformat(f"{time_end_str[:2]}:{time_end_str[2:]}")
            except ValueError:
                return

            # Filtrado de Datos Simulados
            times_simulado = df_simulacion[df_simulacion['trayecto'] == segmento_clave]['minutos_viaje_simulado']

            if times_simulado.empty:
                print(f"Advertencia: No hay datos simulados para {segmento_clave}.")
                return

            # Filtrado de Datos Originales (si existen y tienen la columna correcta)
            times_original_segmento = pd.Series(dtype=float)
            
            # NOTA: Asumo que la columna de tiempo real se llama 'minutos_viaje' en data_clean.csv
            # Si tiene otro nombre (ej. 'duracion', 'tiempo_real'), cámbialo aquí.
            columna_tiempo_real = 'minutos_viaje' 

            if df_original is not None and columna_tiempo_real in df_original.columns:
                pax_original_trayecto = df_original[df_original['trayecto'] == trayecto_original].copy()
                times_original_segmento = pax_original_trayecto.loc[
                    (pax_original_trayecto['hora_salida'] >= time_start) & 
                    (pax_original_trayecto['hora_salida'] < time_end)
                ][columna_tiempo_real].dropna()

            # Generación del Histograma
            plt.figure(figsize=(10, 6))

            # Definir bins basados en tiempos (minutos)
            max_time = times_simulado.max()
            if not times_original_segmento.empty:
                max_time = max(max_time, times_original_segmento.max())
            
            bins = np.linspace(0, max_time + 5, 20) # Bins automáticos para tiempo

            # Histograma Originales
            if not times_original_segmento.empty:
                plt.hist(times_original_segmento, bins=bins, density=True, alpha=0.6, 
                         label='Histórico Real', color='skyblue', edgecolor='black')

            # Histograma Simulados
            plt.hist(times_simulado, bins=bins, density=True, alpha=0.4, 
                     label='Simulación (Time Models)', color='green', edgecolor='black')

            plt.title(f'Distribución de Tiempos de Viaje: {segmento_clave}', fontsize=16)
            plt.xlabel('Minutos de Viaje', fontsize=12)
            plt.ylabel('Densidad', fontsize=12)
            plt.legend()
            plt.grid(axis='y', alpha=0.5)
            plt.show()

        # Bucle para Imprimir
        if df_simulacion is not None:
            todos_los_segmentos = df_simulacion['trayecto'].unique()
            print(f"Iniciando generación de {len(todos_los_segmentos)} histogramas de tiempo...")
            for segmento in todos_los_segmentos:
                graficar_histograma_tiempo_comparativo(segmento, df_simulacion, df_original)