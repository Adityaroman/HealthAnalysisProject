import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(health_file, pop_file):
    '''Load and validate arthritis and population data.'''
    try:
        health_df = pd.read_csv(health_file)
        pop_df = pd.read_csv(pop_file)
        
        required_cols = ['location', 'date', 'month', 'cases', 'severity', 'age_group', 'gender']
        if not all(col in health_df.columns for col in required_cols):
            raise ValueError(f'health_data.csv missing required columns: {required_cols}')
        
        health_df['date'] = pd.to_datetime(health_df['date'])
        health_df['month'] = pd.to_datetime(health_df['month'])
        logging.info('Data loaded successfully')
        return health_df, pop_df
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise

def clean_data(health_df, pop_df):
    '''Clean and preprocess data.'''
    try:
        health_df = health_df.dropna()
        pop_df = pop_df.dropna()
        
        df = health_df.groupby(['location', 'month', 'age_group', 'gender'])['cases'].sum().reset_index()
        df = df.merge(pop_df, on='location', how='left')
        df['cases_per_capita'] = (df['cases'] / df['population'] * 100000).round(2)
        logging.info('Data cleaned successfully')
        return df
    except Exception as e:
        logging.error(f'Error cleaning data: {e}')
        raise

def analyze_trends(df):
    '''Analyze arthritis case trends over time.'''
    try:
        trend_data = df.pivot_table(values='cases', index='month', columns='location', aggfunc='sum', fill_value=0)
        trend_data.index = trend_data.index.strftime('%Y-%m-01')
        logging.info('Trends analyzed')
        return trend_data
    except Exception as e:
        logging.error(f'Error analyzing trends: {e}')
        raise

def identify_high_risk(df):
    '''Identify high-risk areas based on cases per capita.'''
    try:
        risk_data = df.groupby('location')['cases_per_capita'].mean().reset_index()
        risk_data = risk_data.sort_values(by='cases_per_capita', ascending=False)
        logging.info('High-risk areas identified')
        return risk_data
    except Exception as e:
        logging.error(f'Error identifying high-risk areas: {e}')
        raise

def analyze_severity(health_df):
    '''Analyze case severity distribution by state.'''
    try:
        severity_data = health_df.groupby(['location', 'month', 'severity'])['cases'].sum().reset_index()
        severity_data['month'] = severity_data['month'].dt.strftime('%Y-%m-01')
        logging.info('Severity data analyzed')
        return severity_data
    except Exception as e:
        logging.error(f'Error analyzing severity: {e}')
        raise

def analyze_demographics(df):
    '''Analyze cases by age group and gender.'''
    try:
        demo_data = df.groupby(['location', 'month', 'age_group', 'gender'])['cases'].sum().reset_index()
        demo_data['month'] = demo_data['month'].dt.strftime('%Y-%m-01')
        logging.info('Demographic data analyzed')
        return demo_data
    except Exception as e:
        logging.error(f'Error analyzing demographics: {e}')
        raise

def forecast_cases(df, forecast_months=3):
    '''Forecast arthritis cases using Random Forest with focus on high-case states.'''
    try:
        forecasts = {}
        metrics = []
        
        total_cases = df.groupby('location')['cases'].sum().sort_values(ascending=False)
        high_case_locations = total_cases.head(5).index.tolist()
        logging.info(f'High-case states: {high_case_locations}')
        
        for location in df['location'].unique():
            loc_df = df[df['location'] == location].copy()
            loc_df['months'] = (loc_df['month'] - loc_df['month'].min()).dt.days // 30
            
            X = loc_df[['months']].values
            y = loc_df['cases'].values
            
            if len(X) < 2:
                continue
            
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics.append({
                'location': location,
                'mse': mse,
                'r2': r2,
                'high_case': location in high_case_locations
            })
            
            last_month = loc_df['month'].max()
            future_months = np.array([loc_df['months'].max() + i for i in range(1, forecast_months + 1)]).reshape(-1, 1)
            future_cases = model.predict(future_months)
            future_dates = [last_month + timedelta(days=30 * i) for i in range(1, forecast_months + 1)]
            
            forecasts[location] = pd.DataFrame({
                'month': [d.strftime('%Y-%m-01') for d in future_dates],
                'forecasted_cases': future_cases.round(),
                'high_case': location in high_case_locations
            })
        
        forecast_df = pd.concat(forecasts, names=['location']).reset_index()
        logging.info('Forecasting complete')
        return forecast_df, pd.DataFrame(metrics)
    except Exception as e:
        logging.error(f'Error forecasting cases: {e}')
        raise

def main():
    '''Main function to run analysis and save data.'''
    try:
        health_df, pop_df = load_data('health_data.csv', 'population_data.csv')
        df = clean_data(health_df, pop_df)
        
        trend_data = analyze_trends(df)
        trend_data.to_csv('trend_data.csv', index=True)
        
        risk_data = identify_high_risk(df)
        risk_data.to_csv('risk_data.csv', index=False)
        
        severity_data = analyze_severity(health_df)
        severity_data.to_csv('severity_data.csv', index=False)
        
        demo_data = analyze_demographics(df)
        demo_data.to_csv('demographic_data.csv', index=False)
        
        forecast_df, metrics_df = forecast_cases(df)
        forecast_df.to_csv('forecast_data.csv', index=False)
        metrics_df.to_csv('model_metrics.csv', index=False)
        
        logging.info('Analysis complete. Data saved.')
    except Exception as e:
        logging.error(f'Error in analysis: {e}')

if __name__ == '__main__':
    main()