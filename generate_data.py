import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_health_data(num_rows=5000, num_locations=10, start_date='2025-01-01', months=12):
    '''Generate synthetic arthritis case data for Indian states.'''
    try:
        locations = [
            'Maharashtra', 'Uttar Pradesh', 'Tamil Nadu', 'West Bengal', 'Karnataka',
            'Andhra Pradesh', 'Gujarat', 'Rajasthan', 'Kerala', 'Madhya Pradesh'
        ][:num_locations]
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [start + timedelta(days=random.randint(0, months * 30)) for _ in range(num_rows)]
        
        cases = np.random.randint(20, 500, size=num_rows)
        severity = np.random.randint(1, 11, size=num_rows)
        age_groups = random.choices(['18-40', '41-60', '61+'], weights=[0.3, 0.4, 0.3], k=num_rows)
        genders = random.choices(['Male', 'Female'], weights=[0.5, 0.5], k=num_rows)
        
        data = {
            'location': random.choices(locations, k=num_rows),
            'date': dates,
            'month': [d.strftime('%Y-%m-01') for d in dates],
            'cases': cases,
            'severity': severity,
            'age_group': age_groups,
            'gender': genders
        }
        df = pd.DataFrame(data)
        
        df = df.sort_values(by='date')
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        logging.info('Health data generated successfully')
        return df
    except Exception as e:
        logging.error(f'Error generating health data: {e}')
        raise

def generate_population_data(locations):
    '''Generate population data for Indian states.'''
    try:
        population_ranges = {
            'Maharashtra': 112_000_000,
            'Uttar Pradesh': 199_000_000,
            'Tamil Nadu': 72_000_000,
            'West Bengal': 91_000_000,
            'Karnataka': 61_000_000,
            'Andhra Pradesh': 49_000_000,
            'Gujarat': 60_000_000,
            'Rajasthan': 68_000_000,
            'Kerala': 33_000_000,
            'Madhya Pradesh': 72_000_000
        }
        populations = {loc: population_ranges[loc] + random.randint(-2_000_000, 2_000_000) for loc in locations}
        df = pd.DataFrame({
            'location': populations.keys(),
            'population': populations.values()
        })
        logging.info('Population data generated successfully')
        return df
    except Exception as e:
        logging.error(f'Error generating population data: {e}')
        raise

def main():
    '''Generate and save health and population data.'''
    try:
        health_df = generate_health_data()
        health_df.to_csv('health_data.csv', index=False)
        logging.info('Saved health_data.csv')
        
        pop_df = generate_population_data(health_df['location'].unique())
        pop_df.to_csv('population_data.csv', index=False)
        logging.info('Saved population_data.csv')
    except Exception as e:
        logging.error(f'Error in main: {e}')

if __name__ == '__main__':
    main()