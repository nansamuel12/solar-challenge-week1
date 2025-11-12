### MoonLight Energy Solutions: Strategic Solar Investment Brief

This brief synthesizes profiling, cleaning, and cross-country comparison of West African solar datasets (Benin, Sierra Leone, Togo) to inform site prioritization and phased investment.

### Methodology
- Profiled and cleaned each country’s dataset (timestamp parsing, irradiance floor, missing value imputation, basic outlier controls).
- Engineered daily and hourly features; computed daily solar energy proxy and solar hours.
- Built composite Solar Potential Score weighting: average daily GHI (40%), average daily solar hours (30%), clear-sky ratio (20%), and temperature proximity to 25°C (10%).
- Compared countries across seasonal and monthly patterns; ranked regions for deployment.

### Key Insights (data-driven in app)
- Relative ranking by Solar Potential Score highlights the primary near-term deployment target.
- Seasonal GHI patterns suggest optimal commissioning windows and O&M staffing needs.
- Clear-sky ratio and humidity patterns inform technology selection (e.g., panel coatings, tracking).
- Temperature distribution informs expected derating and inverter/thermal design constraints.

### Recommendations
1) Prioritize deployments in the top-ranked country; launch grid-tied pilot plants to accelerate yield validation and tariff negotiations.  
2) Phase-in expansion to the second-ranked country for diversification and resilience.  
3) Schedule commissioning in months with top-quartile average GHI to optimize year-one PR.  
4) Prefer trackers in sites with high clear-sky ratios; consider fixed-tilt where diffuse fraction dominates.  
5) Implement continuous performance monitoring; recalibrate ranking quarterly as new data arrives.  

### Next Steps
- Use the Streamlit dashboard (`app/streamlit_app.py`) to interactively validate ranking, export metrics, and share briefs.  
- Run `scripts/data_profiling.py` and `scripts/country_comparison.py` for offline artifacts (cleaned CSVs, plots, JSON metrics).  
- Feed ranking outputs into site feasibility, grid interconnection assessments, and financial models.  


