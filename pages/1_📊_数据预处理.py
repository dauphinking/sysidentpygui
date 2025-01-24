import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, ccf

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from assist.translations import PAGE_TITLES
import plotly.graph_objects as go
import plotly.express as px
from assist.data_preprocessing import DataPreprocessor
import assist.utils as utils

st.set_page_config(
    page_title=PAGE_TITLES["Data_Preprocessing"],
    page_icon="📊",
    layout="wide",
)

st.title("数据预处理")

utils.addlogo()
utils.removemenu()

# 初始化预处理器
preprocessor = DataPreprocessor()
preprocessor.load_data()

# 在文件开头的 import 部分后添加
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# Function to read data with adaptive headers
def read_data(file, is_input=True):
    try:
        # Always print raw file content first
        print("\n=== Raw File Content ===")
        file.seek(0)
        raw_content = file.read().decode('utf-8')
        print(raw_content)
        print("=== End Raw File Content ===\n")
        
        # Reset file pointer for pandas
        file.seek(0)
        df = pd.read_csv(file, header=None, float_precision='high')
        if is_input:
            st.session_state['x_data_df'] = df
        else:
            st.session_state['y_data_df'] = df
        st.session_state['data_loaded'] = True
        return df
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        st.error(f"Error reading file: {str(e)}")
        return None

# Function to create an interactive plot with synchronized selection
def interactive_plot(x_data, y_data):
    # Debug output
    print("\n=== Plot Data Content ===")
    print("x_data:")
    print(x_data.to_string())
    print("\ny_data:")
    print(y_data.to_string())
    print("=== End Plot Data Content ===\n")
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for x data
    for col in x_data.columns:
        fig.add_trace(go.Scatter(
            x=x_data.index,  # Use actual index
            y=x_data[col],
            name=f'Input {col}',
            mode='lines',
            line=dict(width=2)
        ))
    
    # Add trace for y data
    fig.add_trace(go.Scatter(
        x=y_data.index,  # Use actual index
        y=y_data[y_data.columns[0]],
        name='Output',
        mode='lines',
        line=dict(width=2)
    ))

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(r=150),
        xaxis=dict(
            title='Index',
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(title='Values'),
        dragmode='select',
        selectdirection='h',
        hovermode='x unified'
    )

    # Add selection callback
    fig.update_layout(
        newselection=dict(
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(255,255,255,0.3)'
        )
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def load_data():
    """加载数据并进行预处理"""
    if "x_data" not in st.session_state or "y_data" not in st.session_state:
        st.warning("请先上传数据文件")
        return None, None
    
    try:
        # 获取输入输出数据
        input_data = st.session_state["x_data"]
        output_data = st.session_state["y_data"]
        
        # 打印数据信息
        st.write("### 数据预览")
        
        # 显示输入数据预览
        st.write("输入数据 (前5行):")
        st.write(input_data.head())
        
        # 显示输出数据预览
        st.write("输出数据 (前5行):")
        st.write(output_data.head())
        
        # 绘制时间序列图
        st.write("### 时间序列图")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制输入数据
        for i in range(input_data.shape[1]):
            ax.plot(input_data.iloc[:, i], label=f'输入 {i+1}')
        
        # 绘制输出数据
        ax.plot(output_data, label='输出', linestyle='--')
        
        ax.set_xlabel('样本')
        ax.set_ylabel('值')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
        # 模型结构选择
        st.write("### 模型结构选择")
        model_type = st.selectbox(
            "选择模型结构",
            ["一阶模型 y(k) = a*y(k-1) + b*u(k-1) + c", 
             "二阶模型 y(k) = a1*y(k-1) + a2*y(k-2) + b1*u(k-1) + b2*u(k-2) + c"]
        )
        
        # 保存模型选择到session state
        st.session_state["model_type"] = model_type
        
        # 相关性分析
        st.write("### 相关性分析")
        max_lag = st.slider("最大延迟阶数", 1, 50, 20)
        
        # 计算自相关和互相关
        acf_y = acf(output_data.iloc[:, 0], nlags=max_lag)
        
        # 绘制相关性分析图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 绘制输出自相关
        ax1.stem(range(len(acf_y)), acf_y)
        ax1.set_title('输出自相关函数')
        ax1.set_xlabel('延迟')
        ax1.set_ylabel('相关系数')
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.axhline(y=1.96/np.sqrt(len(output_data)), color='r', linestyle='--')
        ax1.axhline(y=-1.96/np.sqrt(len(output_data)), color='r', linestyle='--')
        
        # 绘制输入输出互相关
        for i in range(input_data.shape[1]):
            ccf_xy = ccf(input_data.iloc[:, i], output_data.iloc[:, 0], unbiased=True)
            ccf_xy = ccf_xy[max_lag:]  # 只取正向延迟
            ax2.stem(range(len(ccf_xy)), ccf_xy, label=f'输入 {i+1}')
        
        ax2.set_title('输入-输出互相关函数')
        ax2.set_xlabel('延迟')
        ax2.set_ylabel('相关系数')
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.axhline(y=1.96/np.sqrt(len(output_data)), color='r', linestyle='--')
        ax2.axhline(y=-1.96/np.sqrt(len(output_data)), color='r', linestyle='--')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 根据相关性分析给出建议
        st.write("### 模型阶次建议")
        
        # 找出显著的自相关阶次
        significant_acf = np.where(np.abs(acf_y) > 1.96/np.sqrt(len(output_data)))[0]
        st.write(f"建议输出延迟阶次: {len(significant_acf)-1}")  # 减去0阶
        
        # 对每个输入找出显著的互相关阶次
        for i in range(input_data.shape[1]):
            ccf_xy = ccf(input_data.iloc[:, i], output_data.iloc[:, 0], unbiased=True)
            ccf_xy = ccf_xy[max_lag:]  # 只取正向延迟
            significant_ccf = np.where(np.abs(ccf_xy) > 1.96/np.sqrt(len(output_data)))[0]
            if len(significant_ccf) > 0:
                st.write(f"建议输入{i+1}延迟阶次: {significant_ccf[0]}")
            else:
                st.write(f"输入{i+1}没有显著的延迟阶次")
        
        return input_data, output_data
        
    except Exception as e:
        st.error(f"处理数据时出错: {str(e)}")
        return None, None

# 加载数据
input_data, output_data = load_data()

# Check if data is loaded
if "x_data" in st.session_state and "y_data" in st.session_state and \
   st.session_state["x_data"] is not None and st.session_state["y_data"] is not None:
    
    # Create columns for input and output data selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("输入数据")
        if not st.session_state.get('data_loaded', False):
            x_data = read_data(st.session_state["x_data"], is_input=True)
        else:
            x_data = st.session_state.get('x_data_df')
        if x_data is not None:
            st.write("输入数据预览：")
            # Print to terminal first
            print("\n=== Input Data Preview ===")
            print(x_data.to_string())
            print("=== End Input Data Preview ===\n")
            # Display in Streamlit
            st.write(x_data.to_string(header=False, float_format=lambda x: '{:.12f}'.format(x)))
    
    with col2:
        st.subheader("输出数据")
        if not st.session_state.get('data_loaded', False):
            y_data = read_data(st.session_state["y_data"], is_input=False)
        else:
            y_data = st.session_state.get('y_data_df')
        if y_data is not None:
            st.write("输出数据预览：")
            # Print to terminal first
            print("\n=== Output Data Preview ===")
            print(y_data.to_string())
            print("=== End Output Data Preview ===\n")
            # Display in Streamlit
            st.write(y_data.to_string(header=False, float_format=lambda x: '{:.12f}'.format(x)))
    
    if x_data is not None and y_data is not None:
        # For plotting, we need column names but don't modify the original dataframes
        x_plot = x_data.copy()
        y_plot = y_data.copy()
        x_plot.columns = [f'x{i+1}' for i in range(len(x_data.columns))]
        y_plot.columns = [f'y{i+1}' for i in range(len(y_data.columns))]
        # Create and display the interactive plot
        interactive_plot(x_plot, y_plot)
else:
    st.warning("请先在'数据加载'页面上传数据") 