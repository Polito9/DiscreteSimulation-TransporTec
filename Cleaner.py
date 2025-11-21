import pandas as pd
import numpy as np

class Cleaner:

    path_info = {} #Dict to save {unit_name, path}

    def __init__(self):
        pass

    def set_path_info(self, path_info: dict):
        self.path_info = path_info
    
    def process_data(self, min_minutes = 1, max_minutes = 30, export_csv = False)->pd.DataFrame:
        all_df = pd.DataFrame()

        for info in self.path_info.items():
            unit_df = pd.read_excel(info[1], sheet_name=None)

            for name, df in unit_df.items():
                df['unidad'] = info[0]
                #Date format standard
                if(len(name) == 6):
                    df['fecha'] = '20'+name[4:]+'-'+name[2:4]+'-'+name[:2]
                else:
                    df['fecha'] = name[4:]+'-'+name[2:4]+'-'+name[:2]
                all_df = pd.concat([all_df, df])
        
        #Selecting columns
        interest_cols = ['inicio', 'final', 'Usuarios', 'hora_salida', 'fecha', 'unidad']
        all_df = all_df.rename(columns={'Punto de inicio':'inicio', 'Punto final ':'final','Hora de salida':'hora_salida', 'Descenso CIEE':'CIEE_desc', 'Descenso Acceso 10':'ACC10_desc'})
        all_df = all_df[interest_cols]

        #Uppercase
        all_df['inicio'] = all_df['inicio'].str.upper()
        all_df['final'] = all_df['final'].str.upper()

        #Null values
        all_df = all_df.dropna(subset=['fecha', 'hora_salida'])

        #Mispelled errors in hours like: 13.42
        all_df['hora_salida'] = all_df['hora_salida'].astype(str).str.replace('.', ':', regex=False)

        #Assuming that there is a mispelled in A1 with A2
        all_df['inicio'] = all_df['inicio'].astype(str).str.replace('A1', 'A2', regex=False)
        all_df['final'] = all_df['final'].astype(str).str.replace('A1', 'A2', regex=False)

        #Spaces
        all_df['hora_salida'] = all_df['hora_salida'].astype(str).str.strip() 
        all_df['inicio'] = all_df['inicio'].astype(str).str.strip()
        all_df['final'] = all_df['final'].astype(str).str.strip()

        #Hour standard format
        all_df['hora_salida'] = all_df['hora_salida'].astype(str).str.slice(0, 5)

        #Adding a timestamp with date and hour
        all_df['timestamp'] = pd.to_datetime(all_df['fecha'].astype(str) + ' ' + all_df['hora_salida'].astype(str), errors='coerce')
        all_df = all_df.dropna(subset=['timestamp'])
        all_df

        #Sorting values
        all_df = all_df.sort_values(by=['fecha', 'unidad', 'timestamp'])
        
        #Calculating the delta
        all_df['siguiente_salida'] = all_df.groupby(['fecha', 'unidad'])['timestamp'].shift(-1)
        all_df['siguiente_inicio'] = all_df.groupby(['fecha', 'unidad'])['inicio'].shift(-1)

        valid_connection = (all_df['final'] == all_df['siguiente_inicio'])

        all_df['delta_tiempo'] = np.where(
            valid_connection, 
            all_df['siguiente_salida'] - all_df['timestamp'], 
            pd.NaT # Not a Time
        )
        all_df['delta_tiempo'] = pd.to_timedelta(all_df['delta_tiempo'])
        #To minutes
        all_df['minutos_viaje'] = all_df['delta_tiempo'].dt.total_seconds() / 60.0

        #The NaN (are the last of the days)
        df_clean = all_df.dropna(subset=['minutos_viaje']).copy() 


        df_final = df_clean[(df_clean['minutos_viaje'] < max_minutes) & (df_clean['minutos_viaje'] >= min_minutes)]

        df_final['trayecto'] = df_final['inicio'] + '-'+ df_final['final']
        
        trayectos = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2']
        
        df_final = df_final.loc[df_final['trayecto'].isin(trayectos)]

        if(export_csv):
            df_final.to_csv('data_clean.csv')
        
        return df_final