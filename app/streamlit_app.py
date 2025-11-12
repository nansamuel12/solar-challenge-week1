"""
MoonLight Energy Solutions - Solar Potential Dashboard
Interactive Streamlit app to profile, clean, compare, and rank regions by solar potential.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.data_loader import (
	load_all_countries,
	clean_solar_data,
	profile_data,
	combine_countries,
)
from scripts.country_comparison import calculate_solar_potential_metrics


# ---------- App Config ----------
st.set_page_config(
	page_title="MoonLight Solar Potential Dashboard",
	layout="wide",
	initial_sidebar_state="expanded",
)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_and_clean(data_dir: str = "data") -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
	raw = load_all_countries(data_dir)
	cleaned = {}
	original = {}
	for country, df in raw.items():
		df_clean, df_orig = clean_solar_data(df)
		cleaned[country] = df_clean
		original[country] = df_orig
	return raw, cleaned, original


@st.cache_data(show_spinner=False)
def compute_profiles(cleaned: Dict[str, pd.DataFrame], original: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
	return {country: profile_data(df, country, original.get(country)) for country, df in cleaned.items()}


@st.cache_data(show_spinner=False)
def compute_metrics(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
	metrics = {country: calculate_solar_potential_metrics(df, country) for country, df in cleaned.items()}
	return pd.DataFrame(metrics).T.sort_values("solar_potential_score", ascending=False)


def plot_time_series(df: pd.DataFrame, y_col: str, title: str) -> go.Figure:
	df_sample = df.iloc[::100].copy() if len(df) > 50000 else df
	fig = px.line(
		df_sample,
		x="Timestamp",
		y=y_col,
		title=title,
		render_mode="webgl",
	)
	fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
	return fig


def plot_monthly_averages(df: pd.DataFrame, metric: str, country: str) -> go.Figure:
	monthly = df.groupby("Month")[metric].mean().reindex(range(1, 13)).reset_index()
	monthly.columns = ["Month", metric]
	fig = px.bar(monthly, x="Month", y=metric, title=f"Average {metric} by Month - {country}")
	fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
	return fig


def plot_hourly_profile(df: pd.DataFrame, metric: str, country: str) -> go.Figure:
	hourly = df.groupby("Hour")[metric].mean().reindex(range(0, 24)).reset_index()
	hourly.columns = ["Hour", metric]
	fig = px.line(hourly, x="Hour", y=metric, markers=True, title=f"Average {metric} by Hour - {country}")
	fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
	return fig


def plot_distribution(df: pd.DataFrame, metric: str, country: str) -> go.Figure:
	fig = px.histogram(df, x=metric, nbins=60, title=f"{metric} Distribution - {country}")
	fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
	return fig


def plot_wind_rose(df: pd.DataFrame, country: str) -> go.Figure:
	"""Generate a wind rose plot using plotly."""
	try:
		# Filter valid wind data
		wind_data = df[(df['WD'].notna()) & (df['WS'].notna()) & 
		               (df['WD'] >= 0) & (df['WD'] <= 360) & 
		               (df['WS'] >= 0)].copy()
		
		if len(wind_data) == 0:
			fig = go.Figure()
			fig.add_annotation(text="No valid wind data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
			return fig
		
		# Create speed bins
		speed_bins = [0, 2, 5, 10, 15, 20, 30, 100]
		speed_labels = ['0-2', '2-5', '5-10', '10-15', '15-20', '20-30', '30+']
		wind_data['Speed_Bin'] = pd.cut(wind_data['WS'], bins=speed_bins, labels=speed_labels)
		
		# Create direction bins (16 directions)
		direction_bins = np.linspace(0, 360, 17)
		direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
		                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
		wind_data['Direction_Bin'] = pd.cut(wind_data['WD'], bins=direction_bins, 
		                                   labels=direction_labels, include_lowest=True)
		
		# Convert directions to radians (for polar plot)
		direction_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
		direction_map = {label: angle for label, angle in zip(direction_labels, direction_angles)}
		
		# Create polar figure
		fig = go.Figure()
		
		# Add traces for each speed bin
		for speed_label in speed_labels:
			speed_data = wind_data[wind_data['Speed_Bin'] == speed_label]
			if len(speed_data) > 0:
				direction_counts = speed_data['Direction_Bin'].value_counts().sort_index()
				
				# Map to angles and frequencies
				angles = []
				frequencies = []
				for dir_label in direction_labels:
					angles.append(direction_map[dir_label])
					frequencies.append(direction_counts.get(dir_label, 0))
				
				# Normalize frequencies to percentages
				total = sum(frequencies)
				if total > 0:
					frequencies_pct = [f / total * 100 for f in frequencies]
					# Complete the circle
					angles.append(angles[0])
					frequencies_pct.append(frequencies_pct[0])
					
					fig.add_trace(go.Scatterpolar(
						r=frequencies_pct,
						theta=np.degrees(angles),
						fill='toself',
						name=f'{speed_label} m/s',
						line=dict(width=1)
					))
		
		fig.update_layout(
			polar=dict(
				radialaxis=dict(
					title="Frequency (%)",
					angle=90,
					showline=True,
					showticklabels=True,
					ticks="outside"
				),
				angularaxis=dict(
					direction="clockwise",
					rotation=90,
					thetaunit="degrees"
				)
			),
			title=f"Wind Rose - {country}",
			showlegend=True,
			height=600
		)
		
		return fig
	except Exception as e:
		fig = go.Figure()
		fig.add_annotation(text=f"Error generating wind rose: {str(e)}", 
		                  xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
		return fig


# ---------- Sidebar ----------
st.sidebar.title("MoonLight Solar Dashboard")
data_dir = st.sidebar.text_input("Data directory", value="data", help="Folder containing CSVs")
countries_available = ["Benin", "Sierra Leone", "Togo"]
selected_countries = st.sidebar.multiselect("Select countries", countries_available, default=countries_available)

metric_choice = st.sidebar.selectbox("Primary metric", ["GHI", "DNI", "DHI", "Tamb", "RH", "WS"])
show_raw = st.sidebar.checkbox("Show raw data tables", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use the download buttons in each section to export results.")


# ---------- Load Data ----------
raw_datasets, cleaned_datasets, original_datasets = load_and_clean(data_dir)
cleaned_datasets = {c: df for c, df in cleaned_datasets.items() if c in selected_countries}
raw_datasets = {c: df for c, df in raw_datasets.items() if c in selected_countries}
original_datasets = {c: df for c, df in original_datasets.items() if c in selected_countries}

if not cleaned_datasets:
	st.error("No datasets loaded. Check the data directory and filenames.")
	st.stop()

profiles = compute_profiles(cleaned_datasets, original_datasets)
comparison_df = compute_metrics(cleaned_datasets)


# ---------- Header ----------
st.title("Strategic Solar Investment Insights")
st.subheader("Profiling, Cleaning, Cross-Country Comparison, and Region Ranking")
st.markdown(
	"Identify high-potential regions for solar installation aligned with long-term sustainability goals."
)


# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Country Profiles", "Cross-Country Comparison", "Strategy Recommendation"])

# Country Profiles
with tab1:
	for country in selected_countries:
		if country not in cleaned_datasets:
			continue
		st.markdown(f"### {country}")
		df = cleaned_datasets[country]
		profile = profiles.get(country, {})

		# Data Quality Information
		if 'cleaning_impact' in profile:
			impact = profile['cleaning_impact']
			st.info(f"**Cleaning Impact**: {impact['records_removed']:,} records removed ({impact['records_before']:,} → {impact['records_after']:,}). "
				   f"Missing values: {impact['missing_before']:,} → {impact['missing_after']:,}")
		
		# Columns with >5% nulls
		if 'columns_high_null' in profile and profile['columns_high_null']:
			high_null_cols = profile['columns_high_null']
			st.warning(f"**Columns with >5% nulls**: {', '.join([f'{col} ({pct:.1f}%)' for col, pct in high_null_cols.items()])}")
		else:
			st.success("**Data Quality**: No columns with >5% nulls")
		
		# KPIs
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Records", f"{len(df):,}")
			st.metric("Solar Hours (GHI>10)", f"{int((df['GHI'] > 10).sum()):,}")
		with col2:
			avg_ghi = df["GHI"].mean() if "GHI" in df.columns else np.nan
			st.metric("Avg GHI", f"{avg_ghi:.1f} W/m²")
			st.metric("Avg DNI", f"{df['DNI'].mean():.1f} W/m²" if "DNI" in df.columns else "N/A")
		with col3:
			st.metric("Avg Temp", f"{df['Tamb'].mean():.1f} °C" if "Tamb" in df.columns else "N/A")
			st.metric("Avg Humidity", f"{df['RH'].mean():.1f} %" if "RH" in df.columns else "N/A")
		with col4:
			st.metric("Avg Wind Speed", f"{df['WS'].mean():.1f} m/s" if "WS" in df.columns else "N/A")
			st.metric("Clear-sky ratio", f"{(df['DNI']/(df['GHI']+1e-6)).where((df['DNI']>0)&(df['GHI']>10)).mean():.2f}")
		
		# Data Statistics (df.describe())
		with st.expander("📊 Detailed Statistics (df.describe())", expanded=False):
			if 'describe' in profile and profile['describe']:
				describe_df = pd.DataFrame(profile['describe'])
				st.dataframe(describe_df, use_container_width=True)
			else:
				numeric_cols = df.select_dtypes(include=[np.number]).columns
				if len(numeric_cols) > 0:
					st.dataframe(df[numeric_cols].describe(), use_container_width=True)
		
		# Pre/Post Cleaning Comparison
		if country in original_datasets:
			df_orig = original_datasets[country]
			with st.expander("🔍 Pre/Post Cleaning Comparison", expanded=False):
				comp_col1, comp_col2 = st.columns(2)
				with comp_col1:
					st.plotly_chart(plot_distribution(df_orig, "GHI", f"{country} - Before Cleaning"), use_container_width=True)
				with comp_col2:
					st.plotly_chart(plot_distribution(df, "GHI", f"{country} - After Cleaning"), use_container_width=True)

		# Plots
		c1, c2 = st.columns(2)
		with c1:
			st.plotly_chart(plot_time_series(df, metric_choice, f"{metric_choice} Over Time - {country}"), use_container_width=True)
			st.plotly_chart(plot_hourly_profile(df, "GHI", country), use_container_width=True)
		with c2:
			st.plotly_chart(plot_monthly_averages(df, "GHI", country), use_container_width=True)
			st.plotly_chart(plot_distribution(df, "GHI", country), use_container_width=True)
		
		# Wind Rose Plot
		if 'WD' in df.columns and 'WS' in df.columns:
			with st.expander("🌹 Wind Rose Plot", expanded=False):
				st.plotly_chart(plot_wind_rose(df, country), use_container_width=True)

		if show_raw:
			with st.expander(f"Show data - {country}", expanded=False):
				st.dataframe(df.head(1000))

		st.download_button(
			label=f"Download cleaned data - {country}",
			data=df.to_csv(index=False).encode("utf-8"),
			file_name=f"{country.lower().replace(' ', '_')}_cleaned.csv",
			mime="text/csv",
		)
		st.markdown("---")

# Cross-Country Comparison
with tab2:
	st.markdown("### Comparative Metrics and Rankings")
	st.dataframe(comparison_df.style.background_gradient(cmap="YlGn").format("{:.2f}", na_rep="-"), use_container_width=True)

	# Rank plot
	score_fig = px.bar(
		comparison_df.reset_index(),
		x="index",
		y="solar_potential_score",
		color="index",
		title="Overall Solar Potential Score",
		labels={"index": "Country", "solar_potential_score": "Score"},
	)
	score_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
	st.plotly_chart(score_fig, use_container_width=True)

	# Monthly GHI comparison
	monthly_traces = []
	for country, df in cleaned_datasets.items():
		monthly_avg = df.groupby("Month")["GHI"].mean().reindex(range(1, 13))
		monthly_traces.append(go.Bar(name=country, x=list(range(1, 13)), y=monthly_avg.values))
	monthly_fig = go.Figure(data=monthly_traces)
	monthly_fig.update_layout(
		barmode="group",
		title="Monthly Average GHI by Country",
		xaxis_title="Month",
		yaxis_title="Average GHI (W/m²)",
		margin=dict(l=10, r=10, t=60, b=10),
	)
	st.plotly_chart(monthly_fig, use_container_width=True)

	# Download comparison
	st.download_button(
		label="Download comparison metrics (CSV)",
		data=comparison_df.to_csv().encode("utf-8"),
		file_name="country_comparison.csv",
		mime="text/csv",
	)
	st.download_button(
		label="Download comparison metrics (JSON)",
		data=json.dumps(comparison_df.to_dict(orient="index"), indent=2).encode("utf-8"),
		file_name="country_metrics.json",
		mime="application/json",
	)

# Strategy Recommendation
with tab3:
	st.markdown("### Data-Driven Strategy Recommendation")
	st.write(
		"Recommendations are derived from a composite Solar Potential Score (GHI, solar hours, clear-sky ratio, and temperature proximity to optimal)."
	)

	topline = comparison_df.head(3)[["solar_potential_score", "avg_daily_ghi", "avg_daily_solar_hours", "clear_sky_ratio", "avg_temperature", "annual_solar_energy"]]
	st.dataframe(topline.style.format("{:.2f}"), use_container_width=True)

	# Simple narrative based on ranking
	best_country = comparison_df.index[0]
	st.success(
		f"Based on current data, {best_country} ranks as the highest-potential region for near-term solar deployment."
	)

	reco_points = [
		"Prioritize grid-tied pilot plants in the top-ranked country to accelerate time-to-value.",
		"Phase expansion into the second-ranked country for portfolio diversification and resilience.",
		"Target commissioning windows in months with highest average GHI to optimize early yield.",
		"Favor sites with higher clear-sky ratios and moderate temperatures to reduce performance derating.",
		"Deploy continuous monitoring to validate assumptions and recalibrate investment cadence quarterly.",
	]
	st.markdown("\n".join([f"- {p}" for p in reco_points]))

	# Export a lightweight strategy brief
	brief = {
		"ranking": comparison_df["solar_potential_score"].rank(ascending=False, method="min").astype(int).to_dict(),
		"top_country": best_country,
		"guidelines": reco_points,
	}
	st.download_button(
		label="Download strategy brief (JSON)",
		data=json.dumps(brief, indent=2).encode("utf-8"),
		file_name="strategy_brief.json",
		mime="application/json",
	)


