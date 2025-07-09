import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'])

try:
    trend_data = pd.read_csv('trend_data.csv')
    risk_data = pd.read_csv('risk_data.csv')
    forecast_data = pd.read_csv('forecast_data.csv')
    severity_data = pd.read_csv('severity_data.csv')
    demo_data = pd.read_csv('demographic_data.csv')
    metrics_data = pd.read_csv('model_metrics.csv')
    trend_data['month'] = pd.to_datetime(trend_data['month'], errors='coerce')
    severity_data['month'] = pd.to_datetime(severity_data['month'], errors='coerce')
    demo_data['month'] = pd.to_datetime(demo_data['month'], errors='coerce')
    forecast_data['month'] = pd.to_datetime(forecast_data['month'], errors='coerce')
    logging.info('CSVs loaded successfully')
except Exception as e:
    logging.error(f"Error loading CSVs: {e}")
    trend_data = pd.DataFrame(columns=['month'])
    risk_data = pd.DataFrame(columns=['location', 'cases_per_capita'])
    forecast_data = pd.DataFrame(columns=['location', 'month', 'forecasted_cases', 'high_case'])
    severity_data = pd.DataFrame(columns=['location', 'month', 'severity', 'cases'])
    demo_data = pd.DataFrame(columns=['location', 'month', 'age_group', 'gender', 'cases'])
    metrics_data = pd.DataFrame(columns=['location', 'mse', 'r2', 'high_case'])

def create_line_chart(data, locations, height=400):
    if data.empty or not locations or 'month' not in data.columns:
        logging.warning('No trend data for line chart')
        return go.Figure().update_layout(title='No Trend Data', height=height)
    fig = go.Figure()
    for loc in locations:
        if loc in data.columns:
            fig.add_trace(go.Scatter(
                x=data['month'], y=data[loc], mode='lines+markers', name=loc,
                line=dict(width=3), marker=dict(size=8),
                hovertemplate=f'{loc}<br>Month: %{{x|%Y-%m}}<br>Cases: %{{y}}'
            ))
    fig.update_layout(
        title='Monthly Arthritis Cases by State', xaxis_title='Month', yaxis_title='Cases',
        height=height, margin=dict(l=50, r=50, t=50, b=50),
        yaxis=dict(range=[0, data[locations].max().max() * 1.2 if locations else 0]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly'
    )
    return fig

def create_bar_chart(data, height=400):
    if data.empty or 'cases_per_capita' not in data.columns:
        logging.warning('No risk data for bar chart')
        return go.Figure().update_layout(title='No Risk Data', height=height)
    fig = px.bar(data, x='cases_per_capita', y='location', title='High-Risk States (Cases per 100,000)',
                 color='cases_per_capita', color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis_title='Cases per 100,000', yaxis_title='State',
        height=height, margin=dict(l=50, r=50, t=50, b=50),
        coloraxis_colorbar=dict(title='Cases/100k')
    )
    return fig

def create_heatmap(data, locations, height=400):
    if data.empty or not locations or 'month' not in data.columns:
        logging.warning('No heatmap data')
        return go.Figure().update_layout(title='No Heatmap Data', height=height)
    heatmap_data = data.melt(id_vars=['month'], value_vars=[col for col in data.columns if col in locations], 
                             var_name='location', value_name='cases')
    heatmap_pivot = heatmap_data.pivot_table(values='cases', index='month', columns='location', fill_value=0)
    fig = px.imshow(heatmap_pivot, x=heatmap_pivot.columns, y=heatmap_pivot.index,
                    title='Cases per State by Month', color_continuous_scale='YlOrRd')
    fig.update_layout(
        xaxis_title='State', yaxis_title='Month',
        height=height, margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_pie_chart(data, category, title, height=400):
    if data.empty or category not in data.columns:
        logging.warning(f'No {category} data for pie chart')
        return go.Figure().update_layout(title=f'No {category} Data', height=height)
    fig = px.pie(data, names=category, values='cases', title=title,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(data))])
    fig.update_layout(height=height, margin=dict(l=50, r=50, t=50, b=50))
    return fig

def create_scatter_plot(data, height=400):
    if data.empty or 'severity' not in data.columns or 'cases' not in data.columns:
        logging.warning('No severity data for scatter plot')
        return go.Figure().update_layout(title='No Severity Data', height=height)
    fig = px.scatter(data, x='severity', y='cases', color='location', size='cases',
                     title='Cases vs. Severity by State')
    fig.update_layout(
        xaxis_title='Severity (1-10)', yaxis_title='Cases',
        height=height, margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_box_plot(data, height=400):
    if data.empty or 'cases' not in data.columns:
        logging.warning('No distribution data for box plot')
        return go.Figure().update_layout(title='No Distribution Data', height=height)
    fig = px.box(data, x='location', y='cases', title='Case Distribution by State',
                 color='location', points='outliers')
    fig.update_layout(
        xaxis_title='State', yaxis_title='Cases',
        height=height, margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    return fig

def create_area_chart(data, locations, height=400):
    if data.empty or not locations or 'month' not in data.columns:
        logging.warning('No cumulative data for area chart')
        return go.Figure().update_layout(title='No Cumulative Data', height=height)
    fig = go.Figure()
    for loc in locations:
        if loc in data.columns:
            fig.add_trace(go.Scatter(
                x=data['month'], y=data[loc], mode='lines', name=loc,
                stackgroup='one', fill='tonexty',
                hovertemplate=f'{loc}<br>Month: %{{x|%Y-%m}}<br>Cases: %{{y}}'
            ))
    fig.update_layout(
        title='Cumulative Cases by State', xaxis_title='Month', yaxis_title='Cumulative Cases',
        height=height, margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig

def create_sunburst_chart(data, height=400):
    if data.empty or not all(col in data.columns for col in ['location', 'age_group', 'gender', 'cases']):
        logging.warning('No demographic data for sunburst chart')
        return go.Figure().update_layout(title='No Demographic Data', height=height)
    fig = px.sunburst(data, path=['location', 'age_group', 'gender'], values='cases',
                      title='Demographic Breakdown of Cases')
    fig.update_layout(height=height, margin=dict(l=50, r=50, t=50, b=50))
    return fig

locations = [col for col in trend_data.columns if col != 'month'] if not trend_data.empty else []
start_date = trend_data['month'].min() if not trend_data.empty else datetime(2025, 1, 1)
end_date = trend_data['month'].max() if not trend_data.empty else datetime(2025, 12, 31)

app.layout = html.Div(id='root', className='bg-gray-100 p-4 dark:bg-gray-800', children=[
    html.H1('Arthritis Case Analysis Dashboard', className='text-3xl font-bold text-center mb-6 text-blue-800 dark:text-blue-300'),
    html.Div(className='flex justify-between mb-4', children=[
        html.Button('Toggle Dark Mode', id='theme-toggle', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 dark:bg-blue-700'),
        html.Div(id='error-alert', className='text-red-600 font-semibold')
    ]),
    html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md mb-6', children=[
        html.H2('Filters', className='text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200'),
        html.Div(className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4', children=[
            dcc.Dropdown(
                id='state-filter',
                options=[{'label': 'All States', 'value': 'all'}] + [{'label': loc, 'value': loc} for loc in locations],
                value='all',
                multi=True,
                placeholder='Select States',
                className='border rounded p-2 bg-white dark:bg-gray-600 dark:text-white'
            ),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=start_date,
                max_date_allowed=end_date,
                start_date=start_date,
                end_date=end_date,
                className='border rounded p-2 bg-white dark:bg-gray-600'
            ),
            dcc.Dropdown(
                id='age-group-filter',
                options=[{'label': 'All Ages', 'value': 'all'}] + [{'label': age, 'value': age} for age in demo_data['age_group'].unique() if not demo_data.empty],
                value='all',
                placeholder='Select Age Group',
                className='border rounded p-2 bg-white dark:bg-gray-600 dark:text-white'
            ),
            dcc.Dropdown(
                id='gender-filter',
                options=[{'label': 'All Genders', 'value': 'all'}] + [{'label': gen, 'value': gen} for gen in demo_data['gender'].unique() if not demo_data.empty],
                value='all',
                placeholder='Select Gender',
                className='border rounded p-2 bg-white dark:bg-gray-600 dark:text-white'
            ),
            dcc.Dropdown(
                id='severity-filter',
                options=[{'label': 'All Severities', 'value': 'all'}] + [{'label': f'Severity {i}', 'value': str(i)} for i in range(1, 11)],
                value='all',
                placeholder='Select Severity',
                className='border rounded p-2 bg-white dark:bg-gray-600 dark:text-white'
            ),
            dcc.Checklist(
                id='high-case-filter',
                options=[{'label': 'Show High-Case States Only', 'value': 'high'}],
                value=[],
                className='p-2'
            ),
            html.Button('Reset Filters', id='reset-filters', className='bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 dark:bg-gray-600')
        ])
    ]),
    html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md mb-6', children=[
        html.H2('Summary Statistics', className='text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200'),
        html.Div(id='summary-stats', className='grid grid-cols-1 sm:grid-cols-3 gap-4')
    ]),
    html.Div(className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4', children=[
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Trends Over Time', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='trend-chart', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('High-Risk States', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='risk-chart', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Cases by State and Month', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='heatmap', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Cases by Age Group', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='age-pie', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Cases by Gender', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='gender-pie', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Cases vs. Severity', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='severity-scatter', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Case Distribution', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='box-plot', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Cumulative Cases', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='area-chart', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Demographic Breakdown', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dcc.Graph(id='sunburst-chart', responsive=True, style={'height': '400px'})
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Model Performance', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dash_table.DataTable(
                id='metrics-table',
                data=metrics_data.to_dict('records') if not metrics_data.empty else [],
                columns=[{'name': i, 'id': i} for i in metrics_data.columns] if not metrics_data.empty else [],
                style_table={'overflowX': 'auto', 'maxHeight': '400px'},
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': '14px'},
                style_header={'backgroundColor': '#1e40af', 'color': 'white', 'fontWeight': 'bold'}
            )
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Forecasted Cases', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            dash_table.DataTable(
                id='forecast-table',
                data=forecast_data.to_dict('records') if not forecast_data.empty else [],
                columns=[{'name': i, 'id': i} for i in forecast_data.columns] if not forecast_data.empty else [],
                style_table={'overflowX': 'auto', 'maxHeight': '400px'},
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': '14px'},
                style_header={'backgroundColor': '#1e40af', 'color': 'white', 'fontWeight': 'bold'}
            )
        ]),
        html.Div(className='bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md', children=[
            html.H2('Download Data', className='text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200'),
            html.Button('Download Trend Data', id='download-trend', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 dark:bg-blue-700 mr-2 mb-2'),
            dcc.Download(id='download-trend-data'),
            html.Button('Download Risk Data', id='download-risk', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 dark:bg-blue-700 mr-2 mb-2'),
            dcc.Download(id='download-risk-data'),
            html.Button('Download Forecast Data', id='download-forecast', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 dark:bg-blue-700 mr-2 mb-2'),
            dcc.Download(id='download-forecast-data'),
            html.Button('Download Severity Data', id='download-severity', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 dark:bg-blue-700 mr-2 mb-2'),
            dcc.Download(id='download-severity-data'),
            html.Button('Download Demographic Data', id='download-demographic', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 dark:bg-blue-700 mb-2'),
            dcc.Download(id='download-demographic-data')
        ])
    ]),
    html.Script('''
        document.getElementById('theme-toggle').addEventListener('click', () => {
            const root = document.getElementById('root');
            root.classList.toggle('dark');
            localStorage.setItem('theme', root.classList.contains('dark') ? 'dark' : 'light');
            document.getElementById('theme-toggle').innerText = root.classList.contains('dark') ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        });
        if (localStorage.getItem('theme') === 'dark') {
            document.getElementById('root').classList.add('dark');
            document.getElementById('theme-toggle').innerText = 'Switch to Light Mode';
        }
    '''),
    dcc.Loading(id='loading', type='cube', children=[html.Div(id='loading-output')])
])

@app.callback(
    [
        Output('trend-chart', 'figure'),
        Output('risk-chart', 'figure'),
        Output('heatmap', 'figure'),
        Output('age-pie', 'figure'),
        Output('gender-pie', 'figure'),
        Output('severity-scatter', 'figure'),
        Output('box-plot', 'figure'),
        Output('area-chart', 'figure'),
        Output('sunburst-chart', 'figure'),
        Output('forecast-table', 'data'),
        Output('summary-stats', 'children'),
        Output('error-alert', 'children')
    ],
    [
        Input('state-filter', 'value'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('age-group-filter', 'value'),
        Input('gender-filter', 'value'),
        Input('severity-filter', 'value'),
        Input('high-case-filter', 'value'),
        Input('reset-filters', 'n_clicks')
    ]
)
def update_dashboard(states, start_date, end_date, age_group, gender, severity, high_case, reset_clicks):
    logging.info('Starting dashboard update')
    error_message = ''
    try:
        if reset_clicks:
            logging.info('Resetting filters')
            states = 'all'
            start_date = str(start_date)
            end_date = str(end_date)
            age_group = 'all'
            gender = 'all'
            severity = 'all'
            high_case = []
        
        filtered_trend = trend_data.copy() if not trend_data.empty else pd.DataFrame(columns=['month'])
        filtered_risk = risk_data.copy() if not risk_data.empty else pd.DataFrame(columns=['location', 'cases_per_capita'])
        filtered_severity = severity_data.copy() if not severity_data.empty else pd.DataFrame(columns=['location', 'month', 'severity', 'cases'])
        filtered_demo = demo_data.copy() if not demo_data.empty else pd.DataFrame(columns=['location', 'month', 'age_group', 'gender', 'cases'])
        filtered_forecast = forecast_data.copy() if not forecast_data.empty else pd.DataFrame(columns=['location', 'month', 'forecasted_cases', 'high_case'])
        
        if not filtered_risk.empty and 'cases_per_capita' in filtered_risk.columns:
            filtered_risk['cases_per_capita'] = pd.to_numeric(filtered_risk['cases_per_capita'], errors='coerce')
        if not filtered_severity.empty and 'severity' in filtered_severity.columns:
            filtered_severity['severity'] = pd.to_numeric(filtered_severity['severity'], errors='coerce')
        
        if 'high' in high_case and not filtered_forecast.empty and 'high_case' in filtered_forecast.columns:
            high_case_states = filtered_forecast[filtered_forecast['high_case'] == True]['location'].unique()
            filtered_trend = filtered_trend[['month'] + [col for col in filtered_trend.columns if col in high_case_states]] if 'month' in filtered_trend.columns else filtered_trend
            filtered_risk = filtered_risk[filtered_risk['location'].isin(high_case_states)]
            filtered_severity = filtered_severity[filtered_severity['location'].isin(high_case_states)]
            filtered_demo = filtered_demo[filtered_demo['location'].isin(high_case_states)]
            filtered_forecast = filtered_forecast[filtered_forecast['location'].isin(high_case_states)]
        
        if states != 'all':
            if isinstance(states, str):
                states = [states]
            filtered_trend = filtered_trend[['month'] + [col for col in filtered_trend.columns if col in states]] if 'month' in filtered_trend.columns else filtered_trend
            filtered_risk = filtered_risk[filtered_risk['location'].isin(states)]
            filtered_severity = filtered_severity[filtered_severity['location'].isin(states)]
            filtered_demo = filtered_demo[filtered_demo['location'].isin(states)]
            filtered_forecast = filtered_forecast[filtered_forecast['location'].isin(states)]
        
        if start_date and end_date and not filtered_trend.empty and 'month' in filtered_trend.columns:
            start = pd.to_datetime(start_date, errors='coerce')
            end = pd.to_datetime(end_date, errors='coerce')
            if start is pd.NaT or end is pd.NaT:
                raise ValueError('Invalid date format')
            filtered_trend = filtered_trend[(filtered_trend['month'] >= start) & (filtered_trend['month'] <= end)]
            
            if not filtered_severity.empty and 'month' in filtered_severity.columns:
                filtered_severity = filtered_severity[filtered_severity['month'].between(start, end)]
            if not filtered_demo.empty and 'month' in filtered_demo.columns:
                filtered_demo = filtered_demo[filtered_demo['month'].between(start, end)]
            if not filtered_forecast.empty and 'month' in filtered_forecast.columns:
                filtered_forecast = filtered_forecast[filtered_forecast['month'] <= end]
        
        if age_group != 'all' and not filtered_demo.empty and 'age_group' in filtered_demo.columns:
            filtered_demo = filtered_demo[filtered_demo['age_group'] == age_group]
        if gender != 'all' and not filtered_demo.empty and 'gender' in filtered_demo.columns:
            filtered_demo = filtered_demo[filtered_demo['gender'] == gender]
        if severity != 'all' and not filtered_severity.empty and 'severity' in filtered_severity.columns:
            filtered_severity = filtered_severity[filtered_severity['severity'] == int(severity)]
        
        stats = [
            html.Div(className='bg-gray-100 dark:bg-gray-600 p-4 rounded', children=[
                html.H3('Total Cases', className='text-lg font-semibold text-gray-800 dark:text-gray-200'),
                html.P(f"{int(filtered_demo['cases'].sum()) if not filtered_demo.empty and 'cases' in filtered_demo.columns else 0:,}")
            ]),
            html.Div(className='bg-gray-100 dark:bg-gray-600 p-4 rounded', children=[
                html.H3('Avg Cases per 100k', className='text-lg font-semibold text-gray-800 dark:text-gray-200'),
                html.P(f"{filtered_risk['cases_per_capita'].mean().round(2) if not filtered_risk.empty and 'cases_per_capita' in filtered_risk.columns else 0:.2f}")
            ]),
            html.Div(className='bg-gray-100 dark:bg-gray-600 p-4 rounded', children=[
                html.H3('Avg Severity', className='text-lg font-semibold text-gray-800 dark:text-gray-200'),
                html.P(f"{filtered_severity['severity'].mean().round(2) if not filtered_severity.empty and 'severity' in filtered_severity.columns else 0:.2f}")
            ])
        ]
        
        locations_to_use = locations if states == 'all' else states if isinstance(states, list) else [states]
        trend_fig = create_line_chart(filtered_trend, locations_to_use)
        risk_fig = create_bar_chart(filtered_risk)
        heatmap_fig = create_heatmap(filtered_trend, locations_to_use)
        age_pie = create_pie_chart(filtered_demo, 'age_group', 'Cases by Age Group')
        gender_pie = create_pie_chart(filtered_demo, 'gender', 'Cases by Gender')
        severity_scatter = create_scatter_plot(filtered_severity)
        box_plot = create_box_plot(filtered_demo)
        area_chart = create_area_chart(filtered_trend, locations_to_use)
        sunburst_chart = create_sunburst_chart(filtered_demo)
        
        logging.info('Dashboard update completed')
        return (
            trend_fig, risk_fig, heatmap_fig, age_pie, gender_pie,
            severity_scatter, box_plot, area_chart, sunburst_chart,
            filtered_forecast.to_dict('records') if not filtered_forecast.empty else [],
            stats, ''
        )
    except Exception as e:
        error_message = f"Error updating dashboard: {str(e)}"
        logging.error(error_message)
        empty_fig = go.Figure().update_layout(title='Error', height=400)
        return (
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig, empty_fig,
            [], [html.Div('Error generating statistics', className='text-red-600')], error_message
        )

@app.callback(
    Output('download-trend-data', 'data'),
    Input('download-trend', 'n_clicks'),
    prevent_initial_call=True
)
def download_trend(n_clicks):
    logging.info('Downloading trend data')
    return dcc.send_data_frame(trend_data.to_csv, 'trend_data.csv')

@app.callback(
    Output('download-risk-data', 'data'),
    Input('download-risk', 'n_clicks'),
    prevent_initial_call=True
)
def download_risk(n_clicks):
    logging.info('Downloading risk data')
    return dcc.send_data_frame(risk_data.to_csv, 'risk_data.csv')

@app.callback(
    Output('download-forecast-data', 'data'),
    Input('download-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def download_forecast(n_clicks):
    logging.info('Downloading forecast data')
    return dcc.send_data_frame(forecast_data.to_csv, 'forecast_data.csv')

@app.callback(
    Output('download-severity-data', 'data'),
    Input('download-severity', 'n_clicks'),
    prevent_initial_call=True
)
def download_severity(n_clicks):
    logging.info('Downloading severity data')
    return dcc.send_data_frame(severity_data.to_csv, 'severity_data.csv')

@app.callback(
    Output('download-demographic-data', 'data'),
    Input('download-demographic', 'n_clicks'),
    prevent_initial_call=True
)
def download_demographic(n_clicks):
    logging.info('Downloading demographic data')
    return dcc.send_data_frame(demo_data.to_csv, 'demographic_data.csv')

if __name__ == '__main__':
    app.run(debug=True, port=8052)