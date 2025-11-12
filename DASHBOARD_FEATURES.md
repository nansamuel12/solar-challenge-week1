# MoonLight Solar Analytics Dashboard - Interactive Features Documentation

## 🌞 Dashboard Overview

The enhanced MoonLight Solar Analytics Dashboard is a comprehensive, interactive Streamlit application designed for solar energy data analysis and investment decision-making. The dashboard provides real-time data visualization, statistical analysis, and strategic insights for solar potential assessment across multiple countries.

## 🎛️ Interactive Features

### 1. **Dynamic Data Loading**
- **Data Source Selection**: Choose between loading raw data from `src/` directory or using pre-cleaned data from `data/` directory
- **Country Selection**: Multi-select interface to analyze specific countries (Benin, Sierra Leone, Togo)
- **Real-time Data Processing**: Automatic data cleaning and profiling on load

### 2. **Interactive Controls**
- **Sampling Rate Slider**: Adjust data sampling from 100 to 10,000 points for performance optimization
- **Primary Metric Selector**: Choose main analysis metric (GHI, DNI, DHI, Temperature, Humidity, Wind Speed)
- **Analysis Mode Switch**: Toggle between Overview, Detailed Analysis, Investment Analysis, and Statistical Analysis
- **Advanced Options Panel**: 
  - Daytime GHI threshold configuration
  - Outlier removal toggle
  - Custom parameter settings

### 3. **Multi-Mode Analysis**

#### 📊 Executive Overview Mode
- **Key Performance Indicators**: Real-time metrics cards showing total records, average solar score, maximum GHI, and top country
- **Country Rankings Table**: Interactive, color-coded ranking table with gradient highlighting
- **Solar Potential Gauge**: Visual gauge chart showing top country's performance
- **Key Insights Panel**: Automated insights and recommendations

#### 🔍 Detailed Analysis Mode
- **Country-Specific Analysis**: Deep dive into individual country data
- **Tabbed Interface**: 
  - Time Series Analysis with trend lines
  - Distribution Analysis with boxplots
  - Pattern Analysis with heatmaps
  - Correlation Analysis with interactive matrices
- **Interactive Controls**: Toggle trend lines, seasonal decomposition, and statistical overlays

#### 💰 Investment Analysis Mode
- **ROI Dashboard**: Comprehensive investment metrics visualization
- **Interactive Investment Calculator**: 
  - Plant capacity slider (1-100 MW)
  - Initial cost configuration ($500K-$2M per MW)
  - Electricity price setting ($0.05-$0.20 per kWh)
- **Financial Projections**: Real-time calculation of payback period, annual revenue, and 25-year projections
- **Strategic Recommendations**: Data-driven investment insights

#### 📊 Statistical Analysis Mode
- **Automated Statistical Testing**: Kruskal-Wallis or ANOVA based on data distribution
- **Statistical Significance Testing**: Post-hoc pairwise comparisons
- **Summary Statistics Table**: Comprehensive statistical comparison across countries
- **Distribution Visualizations**: Violin plots and statistical charts

### 4. **Advanced Visualizations**

#### Interactive Time Series
- **Hover-over Details**: Detailed information on mouse hover
- **Zoom and Pan**: Interactive chart navigation
- **Sampling Control**: Adjustable data point sampling for performance
- **Multiple Metrics**: Support for all solar and weather parameters

#### Distribution Comparisons
- **Interactive Boxplots**: Country-by-country comparison
- **Outlier Detection**: Visual identification of data outliers
- **Statistical Overlays**: Mean, median, and quartile indicators
- **Color Coding**: Consistent color scheme across countries

#### Pattern Heatmaps
- **Hourly vs Monthly Patterns**: 2D heatmap visualization
- **Interactive Tooltips**: Detailed values on hover
- **Color Scales**: Optimized color gradients for solar data
- **Seasonal Insights**: Clear visualization of seasonal patterns

#### Correlation Matrices
- **Interactive Heatmap**: Parameter correlation visualization
- **Numerical Overlays**: Exact correlation coefficients
- **Color-Coded Relationships**: Positive/negative correlations
- **Hover Details**: Detailed correlation information

### 5. **Data Export Capabilities**
- **CSV Export**: Download metrics data in CSV format
- **JSON Export**: Export complete analysis results in JSON
- **Report Generation**: Automated report creation
- **Data Refresh**: Clear cache and reload data functionality

## 🎯 User Experience Features

### Responsive Design
- **Wide Layout**: Optimized for desktop and tablet viewing
- **Expandable Sidebar**: Collapsible control panel
- **Mobile-Friendly**: Responsive design elements
- **Professional Styling**: Custom CSS with gradient effects and modern UI

### Performance Optimization
- **Data Caching**: Streamlit's built-in caching for fast loading
- **Sampling Controls**: Adjustable data sampling for large datasets
- **Lazy Loading**: On-demand data processing
- **Memory Management**: Efficient data handling

### Accessibility
- **Clear Labels**: Descriptive titles and axis labels
- **Color Blind Friendly**: Considerate color schemes
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Proper HTML structure

## 📱 Dashboard Screenshots

The dashboard includes comprehensive visualizations:

1. **Country Rankings** - Bar chart showing solar potential scores
2. **Time Series Analysis** - Interactive temporal data visualization
3. **Distribution Boxplots** - Statistical distribution comparisons
4. **GHI Heatmaps** - Hourly vs monthly pattern analysis
5. **Investment Analysis** - ROI and financial metrics dashboard
6. **Statistical Analysis** - Distribution and significance testing
7. **Correlation Matrix** - Parameter relationship visualization
8. **Solar Gauge** - Visual performance indicator

## 🚀 Technical Implementation

### Architecture
- **Modular Design**: Separate functions for each visualization type
- **Data Pipeline**: Efficient data loading and processing
- **Error Handling**: Robust error management and user feedback
- **Configuration Management**: Flexible parameter controls

### Integration
- **Python Scripts**: Seamless integration with data processing modules
- **Real-time Processing**: Dynamic data fetching and analysis
- **API Ready**: Structure supports future API integration
- **Scalable**: Designed for additional countries and metrics

## 📊 Business Value

### Decision Support
- **Data-Driven Insights**: Statistical validation of investment decisions
- **Risk Assessment**: Comprehensive risk analysis tools
- **ROI Analysis**: Financial modeling and projections
- **Market Comparison**: Side-by-side country analysis

### Operational Efficiency
- **Time Savings**: Automated analysis and reporting
- **Accuracy**: Statistical validation of findings
- **Consistency**: Standardized analysis methodology
- **Scalability**: Easy addition of new markets and data sources

## 🎛️ How to Use

1. **Launch Dashboard**: Run `streamlit run app/enhanced_dashboard.py`
2. **Select Data Source**: Choose raw or cleaned data
3. **Configure Analysis**: Set countries, metrics, and analysis mode
4. **Interact with Visualizations**: Use hover, zoom, and filter features
5. **Export Results**: Download data and generate reports
6. **Refresh Data**: Clear cache and reload as needed

## 📈 Future Enhancements

- **Real-time Data Integration**: Live data feeds from monitoring stations
- **Predictive Analytics**: Machine learning-based forecasting
- **Advanced Financial Modeling**: ROI optimization algorithms
- **Multi-language Support**: International language options
- **API Integration**: External data source connections

---

**Dashboard URL**: http://localhost:8502 (when running locally)
**Data Sources**: Benin, Sierra Leone, Togo solar monitoring stations
**Technologies**: Streamlit, Plotly, Pandas, NumPy, SciPy
