"""
MoonLight Energy Solutions - Enhanced Interactive Solar Dashboard
Advanced Streamlit app with interactive features for solar data analysis and visualization.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scripts.data_loader import (
	load_all_countries,
	clean_solar_data,
	profile_data,
	combine_countries,
)
from scripts.country_comparison import (
	calculate_solar_potential_metrics,
	create_summary_statistics_table,
	perform_statistical_tests,
)

# ---------- App Config ----------
st.set_page_config(
	page_title="MoonLight Solar Potential Dashboard",
	layout="wide",
	initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Data Loading Functions ----------
@st.cache_data(show_spinner=False)
def load_processed_data(data_dir: str = "src") -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
	"""Load and process raw data from src directory"""
	raw = load_all_countries(data_dir)
	cleaned = {}
	original = {}
	for country, df in raw.items():
		df_clean, df_orig = clean_solar_data(df)
		cleaned[country] = df_clean
		original[country] = df_orig
	return raw, cleaned, original

@st.cache_data(show_spinner=False)
def load_cleaned_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
	"""Load pre-cleaned data from data directory"""
	cleaned_datasets = {}
	country_files = {
		'Benin': 'benin_cleaned.csv',
		'Sierra Leone': 'sierra_leone_cleaned.csv', 
		'Togo': 'togo_cleaned.csv'
	}
	
	for country, filename in country_files.items():
		file_path = Path(data_dir) / filename
		if file_path.exists():
			df = pd.read_csv(file_path)
			if 'Timestamp' in df.columns:
				df['Timestamp'] = pd.to_datetime(df['Timestamp'])
			cleaned_datasets[country] = df
	
	return cleaned_datasets

@st.cache_data(show_spinner=False)
def compute_metrics(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
	"""Calculate solar potential metrics for all countries"""
	metrics = {country: calculate_solar_potential_metrics(df, country) for country, df in cleaned.items()}
	return pd.DataFrame(metrics).T.sort_values("solar_potential_score", ascending=False)

@st.cache_data(show_spinner=False)
def compute_profiles(cleaned: Dict[str, pd.DataFrame], original: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
	"""Generate data profiles for all countries"""
	return {country: profile_data(df, country, original.get(country)) for country, df in cleaned.items()}

# ---------- Interactive Visualization Functions ----------
def create_interactive_time_series(df: pd.DataFrame, metric: str, country: str, sample_size: int = 1000) -> go.Figure:
	"""Create interactive time series with sampling control"""
	df_sample = df.sample(min(sample_size, len(df))) if len(df) > sample_size else df
	
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=df_sample['Timestamp'],
		y=df_sample[metric],
		mode='lines',
		name=metric,
		line=dict(width=1),
		hovertemplate='<b>%{fullData.name}</b><br>' +
					  'Time: %{x}<br>' +
					  'Value: %{y:.2f} W/m²<br>' +
					  '<extra></extra>'
	))
	
	fig.update_layout(
		title=f"Interactive {metric} Time Series - {country}",
		xaxis_title="Timestamp",
		yaxis_title=f"{metric} (W/m²)",
		hovermode='x unified',
		showlegend=False
	)
	
	return fig

def create_interactive_boxplot(datasets: Dict[str, pd.DataFrame], metric: str) -> go.Figure:
	"""Create interactive boxplot comparison"""
	data_for_boxplot = []
	
	for country, df in datasets.items():
		if metric in df.columns:
			values = df[metric].dropna()
			for val in values:
				data_for_boxplot.append({
					'Country': country,
					'Metric': metric,
					'Value': val
				})
	
	boxplot_df = pd.DataFrame(data_for_boxplot)
	
	fig = go.Figure()
	
	for i, country in enumerate(boxplot_df['Country'].unique()):
		country_data = boxplot_df[boxplot_df['Country'] == country]['Value']
		fig.add_trace(go.Box(
			y=country_data,
			name=country,
			boxpoints='outliers',
			jitter=0.3,
			pointpos=0,
			marker_color=px.colors.qualitative.Set3[i]
		))
	
	fig.update_layout(
		title=f"Interactive {metric} Distribution Comparison",
		yaxis_title=f"{metric} (W/m²)",
		showlegend=True
	)
	
	return fig

def create_side_by_side_boxplots(datasets: Dict[str, pd.DataFrame]) -> go.Figure:
	"""Create side-by-side boxplots comparing GHI, DNI, DHI across all countries"""
	
	fig = make_subplots(
		rows=1, cols=3,
		subplot_titles=('GHI Distribution', 'DNI Distribution', 'DHI Distribution'),
		horizontal_spacing=0.1
	)
	
	metrics = ['GHI', 'DNI', 'DHI']
	colors = px.colors.qualitative.Set3
	
	for col_idx, metric in enumerate(metrics):
		for row_idx, country in enumerate(datasets.keys()):
			if metric in datasets[country].columns:
				country_data = datasets[country][metric].dropna()
				
				fig.add_trace(go.Box(
					y=country_data,
					name=country,
					boxpoints='outliers',
					jitter=0.3,
					pointpos=0,
					marker_color=colors[row_idx],
					showlegend=(col_idx == 0),  # Only show legend for first subplot
					legendgroup=country
				), row=1, col=col_idx + 1)
	
	fig.update_layout(
		title_text="Side-by-Side Boxplots: GHI, DNI, DHI Comparison Across Countries",
		height=500,
		showlegend=True
	)
	
	fig.update_yaxes(title_text="Irradiance (W/m²)", row=1, col=1)
	
	return fig

def create_interactive_heatmap(df: pd.DataFrame, country: str) -> go.Figure:
	"""Create interactive heatmap of hourly vs monthly GHI patterns"""
	if 'GHI' not in df.columns or 'Hour' not in df.columns or 'Month' not in df.columns:
		return go.Figure()
	
	# Create pivot table for heatmap
	heatmap_data = df.groupby(['Month', 'Hour'])['GHI'].mean().reset_index()
	pivot_table = heatmap_data.pivot(index='Hour', columns='Month', values='GHI')
	
	fig = go.Figure(data=go.Heatmap(
		z=pivot_table.values,
		x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
		   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
		y=pivot_table.index,
		colorscale='YlOrRd',
		hovertemplate='Month: %{x}<br>Hour: %{y}<br>GHI: %{z:.1f} W/m²<extra></extra>'
	))
	
	fig.update_layout(
		title=f"Interactive GHI Heatmap - {country}",
		xaxis_title="Month",
		yaxis_title="Hour of Day"
	)
	
	return fig

def create_solar_potential_gauge(score: float, country: str) -> go.Figure:
	"""Create gauge chart for solar potential score"""
	fig = go.Figure(go.Indicator(
		mode = "gauge+number+delta",
		value = score,
		domain = {'x': [0, 1], 'y': [0, 1]},
		title = {'text': f"Solar Potential Score - {country}"},
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
	
	fig.update_layout(height=400)
	return fig

def create_correlation_matrix(df: pd.DataFrame, country: str) -> go.Figure:
	"""Create interactive correlation matrix"""
	# Select relevant numeric columns
	numeric_cols = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'WSgust', 'BP']
	available_cols = [col for col in numeric_cols if col in df.columns]
	
	if len(available_cols) < 2:
		return go.Figure()
	
	corr_matrix = df[available_cols].corr()
	
	fig = go.Figure(data=go.Heatmap(
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
	
	fig.update_layout(
		title=f"Correlation Matrix - {country}",
		width=600,
		height=500
	)
	
	return fig

# ---------- Advanced Analytics Functions ----------
def create_solar_efficiency_analysis(df: pd.DataFrame, country: str) -> Dict:
	"""Analyze solar efficiency patterns"""
	if 'GHI' not in df.columns:
		return {}
	
	# Calculate efficiency metrics
	hourly_efficiency = df.groupby('Hour')['GHI'].agg(['mean', 'std']).reset_index()
	monthly_efficiency = df.groupby('Month')['GHI'].agg(['mean', 'std']).reset_index()
	
	# Identify peak hours
	peak_hours = hourly_efficiency.nlargest(3, 'mean')['Hour'].tolist()
	
	# Calculate clear sky efficiency
	if 'DNI' in df.columns and 'GHI' in df.columns:
		clear_sky_ratio = (df['DNI'] / (df['GHI'] + 1e-6)).mean()
	else:
		clear_sky_ratio = 0
	
	return {
		'peak_hours': peak_hours,
		'clear_sky_ratio': clear_sky_ratio,
		'hourly_efficiency': hourly_efficiency,
		'monthly_efficiency': monthly_efficiency
	}

def create_investment_roi_analysis(metrics_df: pd.DataFrame) -> go.Figure:
	"""Create ROI analysis visualization"""
	# Simulate ROI calculation based on solar metrics
	roi_data = []
	for country in metrics_df.index:
		score = metrics_df.loc[country, 'solar_potential_score']
		annual_energy = metrics_df.loc[country, 'annual_solar_energy']
		
		# Simulated ROI calculation (simplified)
		initial_investment = 1000000  # $1M
		annual_revenue = annual_energy * 0.1  # $0.1 per kWh
		roi_years = initial_investment / annual_revenue if annual_revenue > 0 else 0
		
		roi_data.append({
			'Country': country,
			'Solar Score': score,
			'Annual Energy (kWh/m²)': annual_energy,
			'ROI Years': roi_years,
			'Annual Revenue ($)': annual_revenue * 1000  # Scale for 1MW plant
		})
	
	roi_df = pd.DataFrame(roi_data)
	
	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=('Solar Potential Score', 'Annual Energy Production', 
					   'ROI Payback Period', 'Projected Annual Revenue'),
		specs=[[{"type": "bar"}, {"type": "bar"}],
			   [{"type": "bar"}, {"type": "bar"}]]
	)
	
	# Solar Score
	fig.add_trace(
		go.Bar(x=roi_df['Country'], y=roi_df['Solar Score'], name='Solar Score'),
		row=1, col=1
	)
	
	# Annual Energy
	fig.add_trace(
		go.Bar(x=roi_df['Country'], y=roi_df['Annual Energy (kWh/m²)'], name='Annual Energy'),
		row=1, col=2
	)
	
	# ROI Years
	fig.add_trace(
		go.Bar(x=roi_df['Country'], y=roi_df['ROI Years'], name='ROI Years'),
		row=2, col=1
	)
	
	# Annual Revenue
	fig.add_trace(
		go.Bar(x=roi_df['Country'], y=roi_df['Annual Revenue ($)'], name='Annual Revenue'),
		row=2, col=2
	)
	
	fig.update_layout(
		title_text="Investment Analysis Dashboard",
		showlegend=False,
		height=600
	)
	
	return fig

# ---------- Main App ----------
def main():
	# Header
	st.markdown('<h1 class="main-header">🌞 MoonLight Solar Analytics Dashboard</h1>', 
				unsafe_allow_html=True)
	st.markdown("---")
	
	# Sidebar Controls
	st.sidebar.title("🎛️ Dashboard Controls")
	
	# Data Source Selection
	data_source = st.sidebar.radio(
		"📊 Data Source",
		["Load Raw Data", "Use Cleaned Data"],
		help="Choose to process raw data or use pre-cleaned datasets"
	)
	
	# Country Selection
	countries_available = ["Benin", "Sierra Leone", "Togo"]
	selected_countries = st.sidebar.multiselect(
		"🌍 Select Countries",
		countries_available,
		default=countries_available,
		help="Choose countries to analyze"
	)
	
	# Interactive Controls
	st.sidebar.markdown("### ⚙️ Analysis Controls")
	
	# Time Range Slider
	time_range = st.sidebar.slider(
		"📅 Data Sampling Rate",
		min_value=100,
		max_value=10000,
		value=1000,
		step=100,
		help="Number of data points to sample for visualizations"
	)
	
	# Metric Selection
	primary_metric = st.sidebar.selectbox(
		"📈 Primary Metric",
		["GHI", "DNI", "DHI", "Tamb", "RH", "WS"],
		help="Main metric for analysis"
	)
	
	# Analysis Mode
	analysis_mode = st.sidebar.selectbox(
		"🔍 Analysis Mode",
		["Overview", "Detailed Analysis", "Investment Analysis", "Statistical Analysis"],
		help="Choose the type of analysis to perform"
	)
	
	# Advanced Options
	show_advanced = st.sidebar.checkbox("🔧 Show Advanced Options", value=False)
	
	if show_advanced:
		st.sidebar.markdown("### 🎛️ Advanced Settings")
		threshold_ghi = st.sidebar.slider(
			"☀️ Daytime GHI Threshold",
			min_value=0,
			max_value=100,
			value=10,
			help="Minimum GHI to consider as daytime"
		)
		
		outlier_removal = st.sidebar.checkbox(
			"📊 Remove Outliers",
			value=False,
			help="Apply outlier removal to visualizations"
		)
	
	# Load Data
	with st.spinner("Loading and processing data..."):
		if data_source == "Load Raw Data":
			raw_datasets, cleaned_datasets, original_datasets = load_processed_data()
		else:
			cleaned_datasets = load_cleaned_data()
			original_datasets = {}
			raw_datasets = {}
		
		# Filter selected countries
		cleaned_datasets = {c: df for c, df in cleaned_datasets.items() if c in selected_countries}
		original_datasets = {c: df for c, df in original_datasets.items() if c in selected_countries}
		raw_datasets = {c: df for c, df in raw_datasets.items() if c in selected_countries}
	
	if not cleaned_datasets:
		st.error("❌ No datasets loaded. Please check your selections.")
		return
	
	# Compute metrics and profiles
	metrics_df = compute_metrics(cleaned_datasets)
	profiles = compute_profiles(cleaned_datasets, original_datasets)
	
	# Main Content Area
	if analysis_mode == "Overview":
		st.markdown("## 📊 Executive Overview")
		
		# Key Metrics Cards
		col1, col2, col3, col4 = st.columns(4)
		
		with col1:
			total_records = sum(len(df) for df in cleaned_datasets.values())
			st.metric("📊 Total Records", f"{total_records:,}")
		
		with col2:
			avg_score = metrics_df['solar_potential_score'].mean()
			st.metric("⭐ Avg Solar Score", f"{avg_score:.2f}")
		
		with col3:
			max_ghi = max(df['GHI'].max() for df in cleaned_datasets.values() if 'GHI' in df.columns)
			st.metric("☀️ Max GHI", f"{max_ghi:.1f} W/m²")
		
		with col4:
			top_country = metrics_df.index[0]
			st.metric("🏆 Top Country", top_country)
		
		# Country Rankings
		st.markdown("### 🏆 Country Rankings")
		
		ranking_cols = st.columns([3, 2, 2])
		with ranking_cols[0]:
			st.dataframe(
				metrics_df[['solar_potential_score', 'avg_daily_ghi', 'annual_solar_energy']]
				.style.background_gradient(cmap='YlGn')
				.format("{:.2f}"),
				use_container_width=True
			)
		
		with ranking_cols[1]:
			# Solar Potential Gauge for top country
			top_score = metrics_df.iloc[0]['solar_potential_score']
			fig_gauge = create_solar_potential_gauge(top_score, metrics_df.index[0])
			st.plotly_chart(fig_gauge, use_container_width=True)
		
		with ranking_cols[2]:
			# Key Insights with strategic findings
			st.markdown("### 💡 Key Insights")
			
			# Calculate strategic insights
			top_country = metrics_df.index[0]
			top_yield = metrics_df.iloc[0]['annual_solar_energy']
			
			# Find country with lowest volatility (lowest std dev)
			lowest_volatility = None
			min_std = float('inf')
			for country in cleaned_datasets.keys():
				if 'GHI' in cleaned_datasets[country].columns:
					std_dev = cleaned_datasets[country]['GHI'].std()
					if std_dev < min_std:
						min_std = std_dev
						lowest_volatility = country
			
			# Identify operational risk (high humidity or low clear sky ratio)
			highest_risk = None
			max_risk_score = 0
			for country in cleaned_datasets.keys():
				df = cleaned_datasets[country]
				risk_score = 0
				if 'RH' in df.columns:
					risk_score += df['RH'].mean() / 100  # Humidity risk
				if 'clear_sky_ratio' in metrics_df.columns:
					risk_score += (1 - metrics_df.loc[country, 'clear_sky_ratio'])  # Clear sky risk
				
				if risk_score > max_risk_score:
					max_risk_score = risk_score
					highest_risk = country
			
			st.markdown(f"""
			<div class="insight-box">
			<strong>🎯 Highest Yield:</strong> {top_country} generates {top_yield:.0f} kWh/m²/year<br>
			<strong>📊 Lowest Volatility:</strong> {lowest_volatility} offers most stable output<br>
			<strong>⚠️ Operational Risk:</strong> {highest_risk} faces weather challenges
			</div>
			""", unsafe_allow_html=True)
		
		# Side-by-Side Boxplots for GHI, DNI, DHI
		st.markdown("### 📊 Side-by-Side Boxplots: Resource Stability Assessment")
		st.markdown("Visual comparison of GHI, DNI, and DHI distributions across countries to assess resource stability and standard deviation.")
		
		fig_boxplots = create_side_by_side_boxplots(cleaned_datasets)
		st.plotly_chart(fig_boxplots, use_container_width=True)
		
		# Statistical Summary for Boxplots
		st.markdown("#### 📈 Statistical Stability Analysis")
		boxplot_stats = []
		
		for country in cleaned_datasets.keys():
			df = cleaned_datasets[country]
			stats_row = {'Country': country}
			
			for metric in ['GHI', 'DNI', 'DHI']:
				if metric in df.columns:
					stats_row[f'{metric}_Mean'] = df[metric].mean()
					stats_row[f'{metric}_Std'] = df[metric].std()
					stats_row[f'{metric}_CV'] = df[metric].std() / df[metric].mean() if df[metric].mean() > 0 else 0
				else:
					stats_row[f'{metric}_Mean'] = 0
					stats_row[f'{metric}_Std'] = 0
					stats_row[f'{metric}_CV'] = 0
			
			boxplot_stats.append(stats_row)
		
		boxplot_df = pd.DataFrame(boxplot_stats)
		st.dataframe(boxplot_df.round(2), use_container_width=True)
	
	elif analysis_mode == "Detailed Analysis":
		st.markdown("## 🔍 Detailed Country Analysis")
		
		# Country selector for detailed view
		detailed_country = st.selectbox("Select country for detailed analysis:", list(cleaned_datasets.keys()))
		
		if detailed_country:
			df = cleaned_datasets[detailed_country]
			profile = profiles.get(detailed_country, {})
			
			# Create tabs for different analysis views
			tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Distribution", "🌡️ Patterns", "🔗 Correlations"])
			
			with tab1:
				st.markdown(f"### Time Series Analysis - {detailed_country}")
				
				# Interactive time series
				fig_ts = create_interactive_time_series(df, primary_metric, detailed_country, time_range)
				st.plotly_chart(fig_ts, use_container_width=True)
				
				# Time series controls
				col1, col2 = st.columns(2)
				with col1:
					show_trend = st.checkbox("Show Trend Line", value=True)
				with col2:
					show_seasonal = st.checkbox("Show Seasonal Decomposition", value=False)
			
			with tab2:
				st.markdown(f"### Distribution Analysis - {detailed_country}")
				
				# Interactive boxplot comparison
				fig_box = create_interactive_boxplot(cleaned_datasets, primary_metric)
				st.plotly_chart(fig_box, use_container_width=True)
				
				# Distribution statistics
				st.markdown("#### 📋 Distribution Statistics")
				if primary_metric in df.columns:
					stats_data = df[primary_metric].describe()
					st.json(stats_data.to_dict())
			
			with tab3:
				st.markdown(f"### Pattern Analysis - {detailed_country}")
				
				# Heatmap
				fig_heatmap = create_interactive_heatmap(df, detailed_country)
				st.plotly_chart(fig_heatmap, use_container_width=True)
				
				# Efficiency analysis
				efficiency_data = create_solar_efficiency_analysis(df, detailed_country)
				if efficiency_data:
					st.markdown("#### ⚡ Efficiency Insights")
					col1, col2 = st.columns(2)
					with col1:
						st.metric("Peak Solar Hours", ", ".join(map(str, efficiency_data['peak_hours'])))
					with col2:
						st.metric("Clear Sky Ratio", f"{efficiency_data['clear_sky_ratio']:.3f}")
			
			with tab4:
				st.markdown(f"### Correlation Analysis - {detailed_country}")
				
				# Correlation matrix
				fig_corr = create_correlation_matrix(df, detailed_country)
				st.plotly_chart(fig_corr, use_container_width=True)
	
	elif analysis_mode == "Investment Analysis":
		st.markdown("## 💰 Investment & ROI Analysis")
		
		# ROI Dashboard
		fig_roi = create_investment_roi_analysis(metrics_df)
		st.plotly_chart(fig_roi, use_container_width=True)
		
		# Investment Recommendations
		st.markdown("### 📋 Investment Recommendations")
		
		rec_cols = st.columns(3)
		with rec_cols[0]:
			st.markdown("""
			<div class="insight-box">
			<h4>🎯 Priority Market</h4>
			<p>Focus immediate resources on the top-performing market for maximum ROI.</p>
			</div>
			""", unsafe_allow_html=True)
		
		with rec_cols[1]:
			st.markdown("""
			<div class="insight-box">
			<h4>📈 Growth Strategy</h4>
			<p>Phase expansion based on market validation and risk assessment.</p>
			</div>
			""", unsafe_allow_html=True)
		
		with rec_cols[2]:
			st.markdown("""
			<div class="insight-box">
			<h4>⚡ Optimization</h4>
			<p>Target peak performance periods for maximum energy yield.</p>
			</div>
			""", unsafe_allow_html=True)
		
		# Cost-Benefit Analysis
		st.markdown("### 💵 Cost-Benefit Analysis")
		
		# Interactive cost calculator
		st.markdown("#### Investment Calculator")
		col1, col2, col3 = st.columns(3)
		
		with col1:
			plant_capacity = st.slider("Plant Capacity (MW)", 1, 100, 10)
		with col2:
			initial_cost = st.slider("Initial Cost ($/MW)", 500000, 2000000, 1000000, step=50000)
		with col3:
			electricity_price = st.slider("Electricity Price ($/kWh)", 0.05, 0.20, 0.10, step=0.01)
		
		# Calculate ROI
		top_country = metrics_df.index[0]
		annual_energy = metrics_df.loc[top_country, 'annual_solar_energy'] * plant_capacity * 1000  # kWh
		annual_revenue = annual_energy * electricity_price
		total_investment = plant_capacity * initial_cost
		payback_period = total_investment / annual_revenue if annual_revenue > 0 else 0
		
		st.markdown("#### 📊 Investment Projections")
		proj_cols = st.columns(4)
		with proj_cols[0]:
			st.metric("Total Investment", f"${total_investment:,.0f}")
		with proj_cols[1]:
			st.metric("Annual Revenue", f"${annual_revenue:,.0f}")
		with proj_cols[2]:
			st.metric("Payback Period", f"{payback_period:.1f} years")
		with proj_cols[3]:
			st.metric("25-Year Revenue", f"${annual_revenue * 25:,.0f}")
	
	elif analysis_mode == "Statistical Analysis":
		st.markdown("## 📊 Statistical Analysis")
		
		# Perform statistical tests
		stat_results = perform_statistical_tests(cleaned_datasets)
		
		# Display statistical results
		st.markdown("### 🔬 Statistical Test Results")
		
		if stat_results:
			stat_cols = st.columns(3)
			with stat_cols[0]:
				st.metric("Test Used", stat_results['test_name'])
			with stat_cols[1]:
				st.metric("Test Statistic", f"{stat_results['statistic']:.4f}")
			with stat_cols[2]:
				st.metric("P-Value", f"{stat_results['p_value']:.6f}")
			
			# Interpretation
			st.markdown("### 📋 Statistical Interpretation")
			if stat_results['significant']:
				st.success("✅ Significant differences exist between countries!")
				st.markdown("""
				**Implications:**
				- Countries have statistically different solar potential
				- Investment decisions should be data-driven
				- Top performer has significant advantage
				""")
			else:
				st.warning("⚠️ No significant differences found")
				st.markdown("""
				**Implications:**
				- Similar solar potential across countries
				- Consider other factors for investment decisions
				- Focus on operational and market differences
				""")
		
		# Summary Statistics Table
		st.markdown("### 📈 Summary Statistics")
		summary_df = create_summary_statistics_table(cleaned_datasets)
		st.dataframe(summary_df, use_container_width=True)
		
		# Statistical Visualizations
		st.markdown("### 📊 Statistical Visualizations")
		
		# Create violin plots for distribution comparison
		fig_violin = go.Figure()
		
		for country, df in cleaned_datasets.items():
			if primary_metric in df.columns:
				fig_violin.add_trace(go.Violin(
					y=df[primary_metric].dropna(),
					name=country,
					box_visible=True,
					meanline_visible=True
				))
		
		fig_violin.update_layout(
			title=f"Distribution Comparison - {primary_metric}",
			yaxis_title=f"{primary_metric} (W/m²)"
		)
		
		st.plotly_chart(fig_violin, use_container_width=True)
	
	# Footer with export options
	st.markdown("---")
	st.markdown("### 📤 Export Options")
	
	export_cols = st.columns(4)
	
	with export_cols[0]:
		if st.button("📊 Export Metrics (CSV)"):
			csv = metrics_df.to_csv(index=True)
			st.download_button(
				label="Download CSV",
				data=csv,
				file_name="solar_metrics.csv",
				mime="text/csv"
			)
	
	with export_cols[1]:
		if st.button("📋 Export Summary (JSON)"):
			json_data = metrics_df.to_dict(orient='index')
			st.download_button(
				label="Download JSON",
				data=json.dumps(json_data, indent=2),
				file_name="solar_summary.json",
				mime="application/json"
			)
	
	with export_cols[2]:
		if st.button("📈 Generate Report"):
			pass
	
	with export_cols[3]:
		if st.button("🔄 Refresh Data"):
			st.cache_data.clear()
			st.rerun()
	
	# Footer info
	st.markdown("---")
	st.markdown("""
	<div style='text-align: center; color: #666;'>
	<p>🌞 MoonLight Energy Solutions Solar Analytics Dashboard | Powered by Streamlit</p>
	<p>Data Sources: Benin, Sierra Leone, Togo Solar Monitoring Stations</p>
	</div>
	""", unsafe_allow_html=True)

if __name__ == "__main__":
	main()
