"""
Dashboard Feature Demonstration Script
Creates sample visualizations to showcase dashboard capabilities.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# Create sample data for demonstration
def create_sample_data():
    """Generate sample solar data for visualization testing"""
    np.random.seed(42)
    
    # Time series data
    dates = pd.date_range('2022-01-01', periods=8760, freq='H')
    sample_data = pd.DataFrame({
        'Timestamp': dates,
        'GHI': np.random.gamma(2, 100, 8760),
        'DNI': np.random.gamma(1.5, 80, 8760),
        'DHI': np.random.gamma(1.8, 60, 8760),
        'Tamb': np.random.normal(25, 5, 8760),
        'RH': np.random.normal(60, 15, 8760),
        'WS': np.random.exponential(2, 8760),
        'Hour': dates.hour,
        'Month': dates.month,
        'Day': dates.day
    })
    
    # Add realistic patterns
    sample_data['GHI'] = sample_data.apply(
        lambda x: x['GHI'] * max(0, np.cos((x['Hour'] - 12) * np.pi / 12)) if 6 <= x['Hour'] <= 18 else 0,
        axis=1
    )
    
    return sample_data

# Create dashboard screenshots
def create_dashboard_screenshots():
    """Generate sample dashboard visualizations"""
    
    # Create output directory
    output_dir = Path("dashboard_screenshots")
    output_dir.mkdir(exist_ok=True)
    
    # Sample data
    sample_data = create_sample_data()
    
    # 1. Executive Overview - Country Rankings
    fig_ranking = go.Figure()
    countries = ['Benin', 'Togo', 'Sierra Leone']
    scores = [71.57, 70.64, 64.34]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig_ranking.add_trace(go.Bar(
        x=countries,
        y=scores,
        marker_color=colors,
        text=[f'{score:.2f}' for score in scores],
        textposition='auto',
    ))
    
    fig_ranking.update_layout(
        title='🏆 Country Solar Potential Rankings',
        xaxis_title='Country',
        yaxis_title='Solar Potential Score',
        showlegend=False,
        height=500
    )
    
    fig_ranking.write_image(output_dir / "country_rankings.png", width=1200, height=600)
    
    # 2. Time Series Analysis
    fig_timeseries = go.Figure()
    
    # Sample hourly data for one week
    week_data = sample_data.head(168)  # 7 days * 24 hours
    
    fig_timeseries.add_trace(go.Scatter(
        x=week_data['Timestamp'],
        y=week_data['GHI'],
        mode='lines',
        name='GHI',
        line=dict(color='orange', width=2),
        hovertemplate='Time: %{x}<br>GHI: %{y:.1f} W/m²<extra></extra>'
    ))
    
    fig_timeseries.update_layout(
        title='☀️ GHI Time Series - Sample Week',
        xaxis_title='Time',
        yaxis_title='GHI (W/m²)',
        hovermode='x unified',
        height=400
    )
    
    fig_timeseries.write_image(output_dir / "time_series.png", width=1200, height=600)
    
    # 3. Distribution Analysis - Boxplots
    fig_boxplot = go.Figure()
    
    # Generate sample data for each country
    benin_data = np.random.gamma(2, 100, 1000)
    togo_data = np.random.gamma(1.8, 90, 1000)
    sierra_data = np.random.gamma(1.5, 80, 1000)
    
    fig_boxplot.add_trace(go.Box(
        y=benin_data,
        name='Benin',
        marker_color='#2ecc71',
        boxpoints='outliers'
    ))
    
    fig_boxplot.add_trace(go.Box(
        y=togo_data,
        name='Togo',
        marker_color='#3498db',
        boxpoints='outliers'
    ))
    
    fig_boxplot.add_trace(go.Box(
        y=sierra_data,
        name='Sierra Leone',
        marker_color='#e74c3c',
        boxpoints='outliers'
    ))
    
    fig_boxplot.update_layout(
        title='📊 GHI Distribution Comparison',
        yaxis_title='GHI (W/m²)',
        height=500
    )
    
    fig_boxplot.write_image(output_dir / "distribution_boxplot.png", width=1200, height=600)
    
    # 4. Heatmap Analysis
    # Create hourly vs monthly heatmap
    heatmap_data = []
    for month in range(1, 13):
        for hour in range(24):
            # Simulate GHI based on hour and month
            base_ghi = max(0, np.cos((hour - 12) * np.pi / 12))
            seasonal_factor = 1 + 0.3 * np.cos((month - 6) * np.pi / 6)
            ghi_value = base_ghi * seasonal_factor * 200 + np.random.normal(0, 20)
            heatmap_data.append({
                'Month': month,
                'Hour': hour,
                'GHI': max(0, ghi_value)
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    pivot_table = heatmap_df.pivot(index='Hour', columns='Month', values='GHI')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=list(range(24)),
        colorscale='YlOrRd',
        hovertemplate='Month: %{x}<br>Hour: %{y}<br>GHI: %{z:.1f} W/m²<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title='🌡️ GHI Heatmap: Hourly vs Monthly Patterns',
        xaxis_title='Month',
        yaxis_title='Hour of Day',
        height=500
    )
    
    fig_heatmap.write_image(output_dir / "ghi_heatmap.png", width=1200, height=600)
    
    # 5. Investment ROI Analysis
    fig_roi = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Solar Potential Score', 'Annual Energy Production', 
                       'ROI Payback Period', 'Projected Annual Revenue'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Sample ROI data
    roi_data = pd.DataFrame({
        'Country': ['Benin', 'Togo', 'Sierra Leone'],
        'Solar Score': [71.57, 70.64, 64.34],
        'Annual Energy': [2049, 1937, 1588],
        'ROI Years': [4.9, 5.2, 6.3],
        'Annual Revenue': [204900, 193700, 158800]
    })
    
    # Add traces
    fig_roi.add_trace(
        go.Bar(x=roi_data['Country'], y=roi_data['Solar Score'], 
               marker_color='#2ecc71', name='Solar Score'),
        row=1, col=1
    )
    
    fig_roi.add_trace(
        go.Bar(x=roi_data['Country'], y=roi_data['Annual Energy'], 
               marker_color='#3498db', name='Annual Energy'),
        row=1, col=2
    )
    
    fig_roi.add_trace(
        go.Bar(x=roi_data['Country'], y=roi_data['ROI Years'], 
               marker_color='#e74c3c', name='ROI Years'),
        row=2, col=1
    )
    
    fig_roi.add_trace(
        go.Bar(x=roi_data['Country'], y=roi_data['Annual Revenue'], 
               marker_color='#f39c12', name='Annual Revenue'),
        row=2, col=2
    )
    
    fig_roi.update_layout(
        title_text="💰 Investment Analysis Dashboard",
        showlegend=False,
        height=600
    )
    
    fig_roi.write_image(output_dir / "investment_analysis.png", width=1200, height=800)
    
    # 6. Statistical Analysis
    fig_stats = go.Figure()
    
    # Create violin plots for distribution comparison
    fig_stats.add_trace(go.Violin(
        y=benin_data,
        name='Benin',
        box_visible=True,
        meanline_visible=True,
        fillcolor='#2ecc71',
        opacity=0.6
    ))
    
    fig_stats.add_trace(go.Violin(
        y=togo_data,
        name='Togo',
        box_visible=True,
        meanline_visible=True,
        fillcolor='#3498db',
        opacity=0.6
    ))
    
    fig_stats.add_trace(go.Violin(
        y=sierra_data,
        name='Sierra Leone',
        box_visible=True,
        meanline_visible=True,
        fillcolor='#e74c3c',
        opacity=0.6
    ))
    
    fig_stats.update_layout(
        title='📊 Statistical Distribution Analysis - GHI',
        yaxis_title='GHI (W/m²)',
        height=500
    )
    
    fig_stats.write_image(output_dir / "statistical_analysis.png", width=1200, height=600)
    
    # 7. Correlation Matrix
    # Create correlation matrix
    numeric_cols = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS']
    corr_matrix = sample_data[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig_corr.update_layout(
        title='🔗 Correlation Matrix - Solar Parameters',
        width=600,
        height=500
    )
    
    fig_corr.write_image(output_dir / "correlation_matrix.png", width=800, height=600)
    
    # 8. Solar Potential Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = 71.57,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Solar Potential Score - Benin"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 85], 'color': "orange"},
                {'range': [85, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=400, title="🎯 Solar Potential Gauge")
    fig_gauge.write_image(output_dir / "solar_gauge.png", width=800, height=400)
    
    print(f"[SUCCESS] Dashboard screenshots created in {output_dir}/")
    print("[SCREENSHOTS] Available visualizations:")
    for file in output_dir.glob("*.png"):
        print(f"   - {file.name}")

if __name__ == "__main__":
    create_dashboard_screenshots()
