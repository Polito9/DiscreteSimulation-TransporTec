from dataclasses import dataclass, field
from typing import List, Dict
import heapq
import pandas as pd
from Bus import Bus
import numpy as np
from datetime import datetime


@dataclass
class Passenger:
    arrival_time: float
    origin: str

@dataclass
class Stop:
    name: str
    queue: List[Passenger] = field(default_factory=list)

class SimulationEngine:
    def __init__(self, buses: List[Bus], pax_model, time_interval, show_process = False):
        self.buses = buses
        self.pax_model = pax_model
        self.start_str, self.end_str = time_interval
        
        self.route_map = {'A2': 'A2-GLAXO', 'GLAXO': 'GLAXO-DISNEY', 'DISNEY': 'DISNEY-A2'}

        #To modify for new route
        self.distances = {'A2': 1.7, 'GLAXO':0.7, 'DISNEY':1.5}
        self.stops = {k: Stop(name=k) for k in self.route_map.keys()}
        self.events = []
        self.results_pax = [] 
        self.show_process = show_process
        
        #Calculating total seconds duration
        fmt = '%H:%M'
        t1 = datetime.strptime(self.start_str, fmt)
        t2 = datetime.strptime(self.end_str, fmt)
        delta = t2 - t1
        self.total_seconds_interval = delta.total_seconds()

    def setup(self):
        # 1. Schedule Passengers (CORRECTED LOGIC HERE)
        if(self.show_process):
            print("Generating Passengers (Accumulating Inter-arrival Times)...")
        
        for stop_name in self.stops:
            route_key = self.route_map[stop_name]
            
            # This returns a list of floats (minutes) representing DIFFERENCES
            inter_arrivals_min = self.pax_model.generate_simulation(
                target_trayecto=route_key, 
                target_interval=(self.start_str, self.end_str), 
                n_samples=1000
            )
            
            # Accumulate time to get absolute simulation clock time
            current_sim_time = 0.0
            
            for delta_min in inter_arrivals_min:
                # Convert minutes to seconds
                delta_sec = delta_min * 60.0
                
                # Add to current clock
                current_sim_time += delta_sec
                
                # Push event: (Time, Priority=2, Type, Data)
                heapq.heappush(self.events, (current_sim_time, 2, 'PAX_ARRIVAL', stop_name))

        # 2. Schedule Buses
        if(self.show_process):
            print("Scheduling Buses...")
        dest_map = {'A2-GLAXO': 'GLAXO', 'GLAXO-DISNEY': 'DISNEY', 'DISNEY-A2': 'A2'}
        
        for bus in self.buses:
            for _, row in bus.times.iterrows():
                time_drive_min = row['minutos_viaje_simulado']
                arrival_time = row['accumulated_seconds']
                base_route = row['trayecto'].split('_')[0]
                stop_location = dest_map.get(base_route)
                
                if stop_location:
                    # Priority 1 ensures bus processing happens *after* passenger arrival if times are equal
                    heapq.heappush(self.events, (arrival_time, 1, 'BUS_ARRIVAL', (bus, stop_location, time_drive_min)))

    def run(self):
        if(self.show_process):
            print("Running Event Loop...")
        while self.events:
            time, _, type, data = heapq.heappop(self.events)
            
            if(time > self.total_seconds_interval):
                #End the simulation if it exceeds the total time duration
                break
            
            if type == 'PAX_ARRIVAL':
                stop_name = data
                self.stops[stop_name].queue.append(Passenger(time, stop_name))
                if(self.show_process):
                    print(f'Pax arrived at {stop_name} at second {time}')
            elif type == 'BUS_ARRIVAL':
                bus, location, total_min = data
                current_stop = self.stops[location]
                
                # Boarding Logic
                boarding_count = 0
                seats = bus.capacity
                
                while current_stop.queue and seats > 0:
                    pax = current_stop.queue.pop(0) # FIFO
                    
                    # Calculate wait time in minutes
                    wait_sec = time - pax.arrival_time
                    wait_min = wait_sec / 60.0 
                    
                    # Ensure wait time isn't negative (edge case with initial offsets)
                    if wait_min < 0: wait_min = 0 
                    
                    self.results_pax.append({
                        'origin': location,
                        'bus_id': bus.id,
                        'wait_time_min': wait_min,
                        'sim_time': time
                    })
                    seats -= 1
                    boarding_count += 1
                if(self.show_process):
                    print(f'Bus {bus.id} arrived at {location} and charged {boarding_count} passengers at second {time}')

                occupancy = (boarding_count / bus.capacity) * 100
                bus.gas_used.append(bus.caculate_gas(boarding_count, total_min, self.distances[location]))
                bus.total_capacity_used_pct.append(occupancy)
                bus.trips_made += 1

    def get_metrics(self, show_metrics = False):
        df_pax = pd.DataFrame(self.results_pax)
        if df_pax.empty:
            print("No passengers boarded.")
            return pd.DataFrame(), pd.DataFrame()

        if(show_metrics):
            print("\n" + "="*40)
            print("      SIMULATION RESULTS SUMMARY      ")
            print("="*40)

        # General Metrics
            print(f"\n>> GENERAL PASSENGER METRICS")
            print(f"Total Passengers Moved: {len(df_pax)}")
            print(f"Global Avg Waiting Time: {df_pax['wait_time_min'].mean():.2f} min")
            print(f"Global Max Waiting Time: {df_pax['wait_time_min'].max():.2f} min")

            # Metrics By Stop
            print(f"\n>> METRICS BY STOP (ORIGIN)")
        
        stop_stats = df_pax.groupby('origin')['wait_time_min'].agg(
            Avg_Wait_Min='mean',
            Max_Wait_Min='max'
            #Pax_Count='count'
        ).reset_index().round(2)


        stop_stats = pd.concat([stop_stats, pd.DataFrame({'origin': ['General'], 'Avg_Wait_Min':[df_pax['wait_time_min'].mean()], 'Max_Wait_Min':[df_pax['wait_time_min'].max()]})])

        if(show_metrics):
            print(stop_stats.to_string(index=False))

        # Bus Metrics
        if(show_metrics):
            print(f"\n>> BUS UTILIZATION & COST")
        
        bus_stats = []
        for bus in self.buses:
            avg_cap = np.mean(bus.total_capacity_used_pct) if bus.total_capacity_used_pct else 0.0
            total_gas = np.sum(bus.gas_used)
            total_cost = bus.trips_made * bus.cost
            
            bus_stats.append({
                'Bus ID': bus.id,
                'Avg Occupancy': f"{avg_cap:.1f}",
                'Trips Made': bus.trips_made,
                'Total Cost': f"{total_cost:.2f}",
                'Total gas (L)': f"{total_gas:.2f}",
            })

        if(show_metrics):
            print(pd.DataFrame(bus_stats).to_string(index=False))

        return stop_stats, pd.DataFrame(bus_stats)