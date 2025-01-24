import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Function to read, merge, and plot CSV data
def plot_csv_data(x_path, y_path):
    # Read the CSV files
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)
    
    # Merge the data
    df = pd.concat([df_x, df_y], axis=1)
    
    # Generate a sequence for the x-axis
    x_axis = range(len(df))
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, df.iloc[:, 0], label='x')
    plt.plot(x_axis, df.iloc[:, 1], label='y')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('x and y Data Plot')
    plt.legend()
    plt.show()

# Example usage
plot_csv_data('xdata3.csv', 'ydata3.csv') 