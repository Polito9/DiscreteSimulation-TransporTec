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

class Buses_model:
    def __init__(self):
        pass

    def generate_model(self):
        # --- 1. DEFINE MANUAL DATA FOR A2-DISNEY ---
        # Mapping intervals to (min_minutes, max_minutes) based on your Google Maps data
        A2_DISNEY_DATA = {
            '06:00-07:00': (5, 8),
            '07:00-09:30': (5, 9),
            '09:30-13:30': (4, 7),
            '13:30-18:00': (4, 10),
            '18:30-22:15': (4, 6)
        }

        # Importing data
        cleaner = Cleaner.Cleaner()
        cleaner.set_path_info({'7': '07_v2.xlsx', '50':'50.xlsx'})
        df = cleaner.process_data(export_csv=True)

        # Small check
        df['hora_salida'] = pd.to_datetime(df['hora_salida'], format='%H:%M').dt.time
        df = df.dropna(subset=['Usuarios']) 

        # Definition of Segments (Added A2-DISNEY)
        TRAYECTOS = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2', 'A2-DISNEY']
        
        TIME_INTERVALS = [
            ('07:00', '09:30'), 
            ('13:30', '18:00'), 
            ('06:00', '07:00'), 
            ('09:30', '13:30'), 
            ('18:30', '22:15')
        ]

        time_models = {}

        print("Iniciando modelado de la distribución de Tiempos de Viaje...")

        for trayecto in TRAYECTOS:
            # We only filter DF if it's NOT the synthetic route
            if trayecto != 'A2-DISNEY':
                sub_trayecto = df.loc[df['trayecto'] == trayecto]
            
            for time_start_str, time_end_str in TIME_INTERVALS:
                
                model_key = f"{trayecto}_{time_start_str.replace(':', '')}-{time_end_str.replace(':', '')}"
                time_data = [] # List to hold our duration data (real or synthetic)
                
                # --- BRANCH A: SYNTHETIC DATA (A2-DISNEY) ---
                if trayecto == 'A2-DISNEY':
                    # Create a key to look up the manual data
                    interval_key = f"{time_start_str}-{time_end_str}"
                    
                    if interval_key in A2_DISNEY_DATA:
                        min_t, max_t = A2_DISNEY_DATA[interval_key]
                        
                        # 1. Calculate Normal Distribution Parameters for Travel
                        mu = (min_t + max_t) / 2
                        sigma = (max_t - min_t) / 4
                        
                        # 2. Generate 1000 synthetic points
                        n_samples = 1000
                        # Travel time (Normal)
                        synthetic_travel = np.random.normal(mu, sigma, n_samples)
                        # Stop time (Uniform between 0 and 2 minutes)
                        synthetic_stop = np.random.uniform(0, 2, n_samples)
                        
                        # 3. Combine and ensure no negative values (sanity check)
                        total_times = synthetic_travel + synthetic_stop
                        total_times = total_times[total_times > 0] 
                        
                        time_data = total_times.tolist()
                        print(f"   -> Generados datos sintéticos para {model_key} (Mu: {mu:.2f} + Stop)")
                    else:
                        print(f"   -> Advertencia: No hay configuración manual para {model_key}")

                # --- BRANCH B: REAL DATA (EXISTING ROUTES) ---
                else:
                    time_start = time.fromisoformat(time_start_str)
                    time_end = time.fromisoformat(time_end_str)
                    
                    sub_time = sub_trayecto.loc[
                        (sub_trayecto['hora_salida'] >= time_start) & 
                        (sub_trayecto['hora_salida'] < time_end)
                    ]
                    
                    if not sub_time.empty and sub_time['minutos_viaje'].count() > 5:
                        time_data = sub_time['minutos_viaje'].tolist()

                # --- COMMON MODEL TRAINING ---
                if len(time_data) > 5:
                    time_model = EmpiricDistribution()
                    
                    # Use the data (whether real or synthetic) to set the histogram
                    time_model.set_histogram(time_data, n_bins=15) 
                    
                    try:
                        time_model.train(show_process=False)
                        time_models[model_key] = time_model
                        print(f"  -> Modelo de Tiempo {model_key} entrenado.")
                    except Exception as e:
                        print(f"  -> Error al entrenar el modelo de Tiempo {model_key}: {e}")
                        time_models[model_key] = None
                else:
                    time_models[model_key] = None
                    print(f"  -> Advertencia: Sin datos suficientes para modelar {model_key}")

        # Almacenamiento de los Modelos
        with open('time_models.pkl', 'wb') as file:
            pickle.dump(time_models, file)

        print("\n Proceso finalizado. Los modelos de tiempo de viaje han sido guardados en 'time_models.pkl'")

    def generate_simulation(self, n_samples=1000, target_time_interval=None):
        """
        Generates simulation data.
        
        Args:
            n_samples (int): Number of samples to generate.
            target_time_interval (tuple, optional): Tuple ('HH:MM', 'HH:MM') to filter a specific interval.
                                                    Example: ('07:00', '09:30')
        """
        # Definición de Parámetros
        NUM_ITERACIONES = n_samples
        TRAYECTOS = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2', 'A2-DISNEY']
        
        # Lista completa de intervalos
        ALL_TIME_INTERVALS = [
            ('07:00', '09:30'),
            ('13:30', '18:00'),
            ('06:00', '07:00'),
            ('09:30', '13:30'),
            ('18:30', '22:15') # Tramo con lógica especial (sustitución)
        ]

        # --- MODIFICACIÓN: Lógica de filtrado de intervalo ---
        if target_time_interval:
            if target_time_interval in ALL_TIME_INTERVALS:
                TIME_INTERVALS = [target_time_interval]
                #print(f"-> Configuración: Simulando ÚNICAMENTE el intervalo {target_time_interval}")
            else:
                #print(f"-> Advertencia: El intervalo {target_time_interval} no existe en la configuración. Se simularán todos.")
                TIME_INTERVALS = ALL_TIME_INTERVALS
        else:
            TIME_INTERVALS = ALL_TIME_INTERVALS
        # -----------------------------------------------------

        # Clave del segmento usado como proxy para TIEMPO
        PROXY_KEY = 'A2-GLAXO_0600-0700'

        # Carga de Modelos
        try:
            with open('time_models.pkl', 'rb') as file:
                time_models = pickle.load(file)
            #print("Modelos de tiempo cargados correctamente.")
            
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

        #print(f"\nIniciando simulación de Monte Carlo (Solo Tiempos) con {NUM_ITERACIONES} iteraciones...")

        for trayecto in TRAYECTOS:
            for time_start, time_end in TIME_INTERVALS:
                
                # Generar la clave única actual
                model_key = f"{trayecto}_{time_start.replace(':', '')}-{time_end.replace(':', '')}"
                
                time_model = time_models.get(model_key)
                
                # Lógica de selección de modelo
                if not time_model or '1830-2215' in model_key:
                    
                    # Condición específica para el tramo nocturno (18:30 - 22:15)
                    # Esta lógica se mantiene intacta si el usuario pide este intervalo específico
                    if '1830-2215' in model_key:
                        # Usamos el proxy original (06:00-07:00) para el tiempo
                        time_model = MODELO_TIEMPO_SUSTITUTO
                        #print(f"  -> Sustitución TIEMPO en {model_key}: Usando proxy {PROXY_KEY}.")

                    else:
                        # Si es otro segmento diferente que falta
                        #print(f"  -> Advertencia: Saltando segmento {model_key} por falta de modelo.")
                        continue
                
                # Generar muestras usando el modelo
                try:
                    # Generamos tiempos
                    simulated_times = time_model.generate_samples(NUM_ITERACIONES)
                    
                except Exception as e:
                    #print(f"  -> Error durante el muestreo del segmento {model_key}: {e}")
                    continue

                # Almacenar los resultados simulados
                df_segment = pd.DataFrame({
                    'trayecto': model_key,
                    'minutos_viaje_simulado': simulated_times,
                    'iteracion': range(len(simulated_times))
                })
                simulated_data.append(df_segment)
                    
                #if '1830-2215' not in model_key:
                    #print(f"  -> Segmento {model_key} simulado con {len(simulated_times)} muestras.")

                
        # Consolidación y Exportación
        if simulated_data:
            df_simulacion_final = pd.concat(simulated_data, ignore_index=True)
            
            #df_simulacion_final.to_csv('bus_results_times.csv', index=False)
            
            #print("\n=======================================================")
            #print(f"Simulación de Tiempos COMPLETADA.")
            #print(f"Total de registros simulados: {len(df_simulacion_final)}")
            #print(f"Resultados guardados en: {output_file}")
            #print("=======================================================")
            return df_simulacion_final
        else:
            print("\nSimulación finalizada sin generar datos.")
            return pd.DataFrame()
        


    def show_results(self):
        # Carga y Pre-procesamiento Inicial de Datos
        try:
            # Intenta cargar el archivo de simulación de tiempos
            df_simulacion = pd.read_csv('bus_results_times.csv')
        except FileNotFoundError:
            print("Error: No se encontró 'bus_results_times.csv'. Ejecuta la simulación primero.")
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

class Pax_model:
    N_BUSES = 0
    lambdas_saved = {}
    TRAYECTOS = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2']
    TIME_INTERVALS = [
        ('07:00', '09:30'), 
        ('13:30', '18:00'), 
        ('06:00', '07:00'), 
        ('09:30', '13:30'), 
        ('18:30', '22:15') 
    ]

    def __init__(self):
        pass

    def generate_model(self, n_buses = 2):
        cleaner = Cleaner.Cleaner()

        cleaner.set_path_info({'7': '07_v2.xlsx', '50':'50.xlsx'})

        df = cleaner.process_data(export_csv=False)

        df['hora_salida'] = pd.to_datetime(df['hora_salida'], format='%H:%M').dt.time
        df = df.dropna(subset=['Usuarios'])

        #Calculating lambdas per interval and place
        
        self.N_BUSES = n_buses


        REF_START, REF_END = '13:30', '18:00'
        TARGET_START, TARGET_END = '18:30', '22:15'

        for start_time, end_time in self.TIME_INTERVALS:
            sub_df = df.loc[(df['hora_salida'].astype(str) >= start_time) & (df['hora_salida'].astype(str) < end_time)]
            avg_cycle_time = sub_df.groupby('trayecto')['minutos_viaje'].mean().sum()

            for t in self.TRAYECTOS:
                key = f"{t}_{start_time.replace(':', '')}-{end_time.replace(':', '')}"
                lambda_ = 0
                if(start_time == TARGET_START):
                    #special case, 20% less than the ('13:30', '18:00')
                    lambda_ = self.lambdas_saved[f"{t}_{REF_START.replace(':', '')}-{REF_END.replace(':', '')}"]*.8
                else:
                    avg_pax_in_bus = sub_df.loc[sub_df['trayecto'] == t]['Usuarios'].mean()
                    #print(avg_pax_in_bus, f'for key: {key}')
                    # Lambda = avg_pax / (cycle_time/n_buses)
                    lambda_ = avg_pax_in_bus / (avg_cycle_time/self.N_BUSES)

                self.lambdas_saved[key] = lambda_
    
    def generate_simulation(self, target_trayecto=None, target_interval=None, n_samples=10000):
        """
        Simulates passenger arrivals based on a Poisson Process (Exponential distribution).
        
        Args:
            target_trayecto (str, optional): Specific route to filter.
            target_interval (tuple, optional): Specific time interval ('HH:MM', 'HH:MM').
            n_samples (int): Number of arrivals to simulate per segment.
        """
        
        # Asumiendo que es un Proceso de Poisson, los tiempos de llegada se comportan como una exponencial
        data_simulated = []

        # Filtrado de intervalos
        if target_interval:
            # Convertimos a lista para iterar
            intervals_to_process = [target_interval]
        else:
            intervals_to_process = self.TIME_INTERVALS

        # Filtro de trayectos
        if target_trayecto:
            trayectos_to_process = [target_trayecto]
        else:
            trayectos_to_process = self.TRAYECTOS

        #print(f"Iniciando simulación de Pasajeros (Poisson) - Muestras: {n_samples}")

        for start_time, end_time in intervals_to_process:
            for t in trayectos_to_process:
                
                # Generamos la key estándar para guardar los datos
                current_key = f"{t}_{start_time.replace(':', '')}-{end_time.replace(':', '')}"
                
                # Definimos qué key usaremos para BUSCAR la lambda en el diccionario guardado
                lookup_key = current_key
                
                # Si estamos en el horario especial (18:30 - 22:15), usamos el lambda de la mañana (06:00-07:00)
                if '1830-2215' in current_key:
                    # Construimos la key del proxy manteniendo el mismo trayecto 't'
                    proxy_key = f"{t}_0600-0700"
                    #print(f"  -> Aviso: Usando Lambda Proxy ({proxy_key}) para el segmento {current_key}")
                    lookup_key = proxy_key

                # Validación de existencia
                if lookup_key not in self.lambdas_saved:
                    print(f"  -> Error: No se encontró lambda para {lookup_key}. Saltando...")
                    continue

                # Obtenemos la tasa (lambda) del diccionario
                lambda_act = self.lambdas_saved[lookup_key]
                
                # Generamos los datos
                # Optimizamos un poco usando list comprehension en lugar del append en bucle
                try:
                    arrivals = [random.expovariate(lambda_act) for _ in range(n_samples)]
                    data_simulated = arrivals
                except Exception as e:
                    print(f"Error generando datos para {current_key}: {e}")

        if data_simulated:
            #output_file = 'pax_time_arrivals.csv'
            #df_pax_simulated.to_csv(output_file, index=False)
            #print(f"Simulación de Pasajeros completada. Guardado en {output_file}")
            return data_simulated
        else:
            print("No se generaron datos (posiblemente claves faltantes o filtros vacíos).")
            return []
