"""
Visualization utilities for the NPRI Time Series Analysis project.
Handles plotting of time series data, model evaluation results, and forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import os


def save_figure(fig, filename, output_dir='../../reports/figures', dpi=300):
    """
    Save a matplotlib figure to the figures directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename for the saved figure
    output_dir : str, default='../../reports/figures'
        Directory to save figures
    dpi : int, default=300
        Resolution for saving the figure
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {filepath}")


def plot_time_series_by_industry(df, target_col, industries=None, title=None, save=False):
    """
    Plot time series of releases by industry.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time series data
    target_col : str
        Target column to plot (e.g., 'Total_Air_Releases')
    industries : list, default=None
        List of industries to include (default: top 5 by release volume)
    title : str, default=None
        Plot title
    save : bool, default=False
        Whether to save the figure
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    # Create a copy of the dataframe
    plot_df = df.copy()
    
    # If industries not specified, use top 5 by release volume
    if industries is None:
        top_industries = plot_df.groupby('Industry_Sector')[target_col].sum().nlargest(5).index
        plot_df = plot_df[plot_df['Industry_Sector'].isin(top_industries)]
    else:
        plot_df = plot_df[plot_df['Industry_Sector'].isin(industries)]
    
    # Group by year and industry
    grouped_df = plot_df.groupby(['Reporting_Year', 'Industry_Sector'])[target_col].sum().reset_index()
    
    # Create plot title if not provided
    if title is None:
        title = f'{target_col} by Industry Over Time'
    
    # Create interactive plot
    fig = px.line(grouped_df, x='Reporting_Year', y=target_col, color='Industry_Sector',
                 title=title,
                 labels={target_col: f'{target_col} (tonnes)'},
                 template='plotly_white')
    
    # Update layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=f'{target_col} (tonnes)',
        legend_title='Industry Sector',
        hovermode='x unified'
    )
    
    # Save figure if requested
    if save:
        filename = f"{target_col.lower().replace('_', '-')}_by_industry.html"
        output_dir = '../../reports/figures'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Interactive plot saved to {filepath}")
    
    return fig


def plot_model_evaluation(results, substance_name, target_type, save=False):
    """
    Plot model evaluation results including actual vs. predicted values and residuals.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model evaluation results
    substance_name : str
        Name of the substance
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
    save : bool, default=False
        Whether to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing evaluation plots
    """
    # Extract predictions and actual values
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot actual vs. predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'Actual vs. Predicted {target_type} - {substance_name}')
    
    # Calculate and plot residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    # Add metrics text
    metrics = results['metrics']
    metrics_text = f"RMSE: {metrics['rmse']:.3f}\nMAE: {metrics['mae']:.3f}\nR²: {metrics['r2']:.3f}"
    axes[0].annotate(metrics_text, xy=(0.05, 0.9), xycoords='axes fraction',
                     bbox=dict(boxstyle='round', fc='whitesmoke'))
    
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        model_name = results.get('model_name', 'model')
        filename = f"{model_name}_{substance_name}_{target_type.lower()}_evaluation.png"
        save_figure(fig, filename)
    
    return fig


def plot_forecast(historical_data, forecast_data, substance_name, target_type, industry_sector=None, save=False):
    """
    Plot historical data and forecasts.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing historical data
    forecast_data : pandas.DataFrame
        DataFrame containing forecast data
    substance_name : str
        Name of the substance
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
    industry_sector : str, default=None
        Industry sector (if None, shows results for all sectors combined)
    save : bool, default=False
        Whether to save the figure
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    # Filter data for specific industry if provided
    if industry_sector is not None:
        historical_data = historical_data[historical_data['Industry_Sector'] == industry_sector].copy()
    
    # Group historical data by year
    historical_grouped = historical_data.groupby('Reporting_Year')[target_type].sum().reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_grouped['Reporting_Year'],
        y=historical_grouped[target_type],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['Forecast_Year'],
        y=forecast_data['Predicted_Value'],
        mode='lines+markers',
        name='Forecast (2024-2028)',
        line=dict(color='red', dash='dash')
    ))
    
    # Set title and labels
    title_text = f"Historical and Forecasted {target_type} - {substance_name}"
    if industry_sector:
        title_text += f" ({industry_sector})"
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Year',
        yaxis_title=f'{target_type} (tonnes)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Add vertical line separating historical and forecast data
    fig.add_vline(x=2023.5, line_dash="dot", line_color="gray",
                 annotation_text="Forecast Begins", annotation_position="top right")
    
    # Save figure if requested
    if save:
        industry_text = "_" + industry_sector.lower().replace(' ', '_') if industry_sector else ""
        filename = f"{substance_name.lower().replace(' ', '_')}_{target_type.lower()}{industry_text}_forecast.html"
        output_dir = '../../reports/figures'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Interactive forecast plot saved to {filepath}")
    
    return fig


def plot_industry_forecast_comparison(forecasts, target_type, growth_industries=None, decline_industries=None, save=False):
    """
    Create a comparison plot of industry forecasts, highlighting growth and decline industries.
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary containing forecast results by industry
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
    growth_industries : list, default=None
        List of industries with highest growth (default: top 3)
    decline_industries : list, default=None
        List of industries with largest decline (default: top 3)
    save : bool, default=False
        Whether to save the figure
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    # Calculate percentage change for each industry
    industry_changes = {}
    
    for industry, forecast_df in forecasts.items():
        first_value = forecast_df['Predicted_Value'].iloc[0]
        last_value = forecast_df['Predicted_Value'].iloc[-1]
        
        if first_value > 0:
            pct_change = ((last_value - first_value) / first_value) * 100
        else:
            pct_change = 0 if last_value == 0 else 100  # Avoid division by zero
            
        industry_changes[industry] = pct_change
    
    # Sort industries by percentage change
    sorted_industries = sorted(industry_changes.items(), key=lambda x: x[1], reverse=True)
    
    # If growth industries not specified, use top 3
    if growth_industries is None:
        growth_industries = [industry for industry, _ in sorted_industries[:3]]
        
    # If decline industries not specified, use bottom 3
    if decline_industries is None:
        decline_industries = [industry for industry, _ in sorted_industries[-3:]]
    
    # Combine selected industries
    selected_industries = growth_industries + decline_industries
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each selected industry
    for industry in selected_industries:
        forecast_df = forecasts[industry]
        
        # Determine color based on growth or decline
        if industry in growth_industries:
            color = 'green'
            dash = None
        else:
            color = 'red'
            dash = 'dash'
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=forecast_df['Forecast_Year'],
            y=forecast_df['Predicted_Value'],
            mode='lines+markers',
            name=industry,
            line=dict(color=color, dash=dash)
        ))
    
    # Set title and labels
    fig.update_layout(
        title=f"Forecast Comparison: Growth vs. Decline Industries ({target_type})",
        xaxis_title='Year',
        yaxis_title=f'{target_type} (tonnes)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Add annotations for percentage changes
    for industry in selected_industries:
        pct_change = industry_changes[industry]
        direction = "↑" if pct_change >= 0 else "↓"
        
        fig.add_annotation(
            x=forecasts[industry]['Forecast_Year'].iloc[-1],
            y=forecasts[industry]['Predicted_Value'].iloc[-1],
            text=f"{direction} {abs(pct_change):.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="black",
            arrowsize=1,
            arrowwidth=1
        )
    
    # Save figure if requested
    if save:
        filename = f"{target_type.lower()}_industry_comparison_forecast.html"
        output_dir = '../../reports/figures'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Interactive industry comparison plot saved to {filepath}")
    
    return fig


def plot_substance_correlations(df, substances, target_type, save=False):
    """
    Plot correlation heatmap between different substances for a specific release type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing release data
    substances : list
        List of substances to include
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
    save : bool, default=False
        Whether to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing correlation heatmap
    """
    # Pivot data to create substance x substance matrix
    pivot_df = pd.pivot_table(
        df,
        values=target_type,
        index='Reporting_Year',
        columns='Substance_Name',
        aggfunc='sum'
    )
    
    # Filter for selected substances
    pivot_df = pivot_df[substances]
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )
    
    plt.title(f'Substance Correlation for {target_type}')
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        filename = f"{target_type.lower()}_substance_correlation.png"
        save_figure(plt.gcf(), filename)
    
    return plt.gcf()
