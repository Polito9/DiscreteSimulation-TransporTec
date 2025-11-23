import pandas as pd
from Model_Training import Buses_model

class Bus:
    id = ''
    start_second = 0
    start_place = ''
    cost = 0.0
    capacity = 0
    times = pd.DataFrame()
    trips_made = 0
    total_capacity_used_pct = []

    def sort_trajectory_data(self, df, start_place):
        # Create a temporary column to identify the route name without the time
        df['base_trayecto'] = df['trayecto'].apply(lambda x: x.split('_')[0])

        # Define the sort order list based on the start_place
        if start_place == 'GLAXO':
            # Cycle: GLAXO -> DISNEY -> A2 -> GLAXO
            order = ['GLAXO-DISNEY', 'DISNEY-A2', 'A2-GLAXO']
            
        elif start_place == 'A2':
            order = ['A2-GLAXO', 'GLAXO-DISNEY', 'DISNEY-A2']
            
        elif start_place == 'DISNEY':
            # Cycle: DISNEY -> A2 -> GLAXO -> DISNEY
            order = ['DISNEY-A2', 'A2-GLAXO', 'GLAXO-DISNEY']
            
        else:
            raise ValueError("start_place must be 'GLAXO', 'A2', or 'DISNEY'")

        sort_map = {name: i for i, name in enumerate(order)}

        # Map the rank to the dataframe
        df['sort_rank'] = df['base_trayecto'].map(sort_map)

        # Sort by Iteration (primary) and Rank (secondary)
        df_sorted = df.sort_values(by=['iteracion', 'sort_rank'])

        df_sorted['seconds_viaje_simulado'] = df_sorted['minutos_viaje_simulado'] * 60
        df_sorted['accumulated_seconds'] = df_sorted['seconds_viaje_simulado'].cumsum() + self.start_second        
        
        # Clean up helper columns
        df_sorted = df_sorted.drop(columns=['base_trayecto', 'sort_rank', 'seconds_viaje_simulado'])


        return df_sorted

    def __init__(self, id, start_second, start_place, cost, capacity, start_time, end_time, samples = 1000):

        self.id = id
        self.start_second = start_second
        self.start_place = start_place
        self.cost = cost
        self.capacity = capacity

        buses_model = Buses_model()

        data = buses_model.generate_simulation(target_time_interval=(start_time, end_time), n_samples=samples)
        sorted_data = self.sort_trajectory_data(data, self.start_place)
        self.times = sorted_data.reset_index().drop(columns=['index'])
