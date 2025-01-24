import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Function to read, align, and plot data

def interactive_plot(x_path, y_path):
    # Read the CSV files
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)
    
    # Align data on the same x-axis
    df = pd.concat([df_x, df_y], axis=1)
    df['Index'] = range(len(df))
    
    # Create an interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Index'], y=df.iloc[:, 0], mode='lines', name='x'))
    fig.add_trace(go.Scatter(x=df['Index'], y=df.iloc[:, 1], mode='lines', name='y'))
    
    # Update layout
    fig.update_layout(title='Interactive Data Plot', xaxis_title='Index', yaxis_title='Values')
    
    # Show plot
    fig.show()

# Example usage
interactive_plot('xdata3.csv', 'ydata3.csv') 