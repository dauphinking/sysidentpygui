import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# 过滤警告信息
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*label got an empty value.*')

class DataPreprocessor:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        self.processed_input = None
        self.processed_output = None
    
    def load_data(self):
        """从session state获取数据"""
        print("\n=== 开始加载数据 ===")
        if "x_data" in st.session_state:
            print(f"x_data found in session state: {st.session_state['x_data']}")
        if "y_data" in st.session_state:
            print(f"y_data found in session state: {st.session_state['y_data']}")
            
        if "x_data" in st.session_state and st.session_state["x_data"] is not None:
            try:
                # 直接从文件对象读取数据
                data = st.session_state["x_data"].getvalue().decode('utf-8').splitlines()
                self.input_data = pd.DataFrame([float(x) for x in data if x.strip()], columns=['value'])
                print("Input data loaded successfully")
                print("Input data shape:", self.input_data.shape)
                print("Input data head:\n", self.input_data.head())
            except Exception as e:
                print("Error loading input data:", str(e))
                
        if "y_data" in st.session_state and st.session_state["y_data"] is not None:
            try:
                # 直接从文件对象读取数据
                data = st.session_state["y_data"].getvalue().decode('utf-8').splitlines()
                self.output_data = pd.DataFrame([float(x) for x in data if x.strip()], columns=['value'])
                print("Output data loaded successfully")
                print("Output data shape:", self.output_data.shape)
                print("Output data head:\n", self.output_data.head())
            except Exception as e:
                print("Error loading output data:", str(e))
        print("=== 数据加载完成 ===\n")
    
    def show_data_preview(self):
        """显示数据预览和基本分析"""
        print("\n=== 开始数据预览 ===")
        if self.input_data is None:
            print("Warning: input_data is None")
            return
        if self.output_data is None:
            print("Warning: output_data is None")
            return
            
        print(f"Input data shape: {self.input_data.shape}")
        print(f"Output data shape: {self.output_data.shape}")
        
        if self.input_data is not None and self.output_data is not None:
            st.subheader("数据预览和分析")
            
            try:
                # 1. 数据统计
                print("Computing data statistics...")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("输入数据统计：")
                    st.dataframe(self.input_data.describe())
                with col2:
                    st.write("输出数据统计：")
                    st.dataframe(self.output_data.describe())
                
                # 2. 时间序列图
                print("Creating time series plots...")
                fig = make_subplots(rows=2, cols=1,
                                  subplot_titles=('Input Time Series', 'Output Time Series'))
                fig.add_trace(
                    go.Scatter(y=self.input_data.iloc[:,0], name="Input"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=self.output_data.iloc[:,0], name="Output"),
                    row=2, col=1
                )
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. 相关性分析
                print("Starting correlation analysis...")
                st.subheader("相关性分析")
                
                # 添加解释说明
                st.markdown("""
                ### 如何使用相关性分析确定延迟阶数(lag)
                
                #### 自相关函数(ACF)
                - ACF显示输出信号与其自身延迟版本的相关性
                - 当ACF值显著不为零时，表明该延迟对系统有影响
                - 通常选择ACF开始显著下降或趋于零时的延迟作为输出延迟阶数
                
                #### 互相关函数(CCF)
                - CCF显示输入信号对输出信号的影响延迟
                - CCF的峰值表明输入对输出的最强影响出现的延迟
                - 选择CCF开始显著下降前的延迟作为输入延迟阶数
                
                #### 选择建议
                - 较大的延迟会增加模型复杂度
                - 建议从较小的延迟开始（如2-3），根据模型性能逐步调整
                - 如果ACF/CCF在较大延迟处仍显著，可以考虑增加延迟阶数
                """)
                
                max_lag = st.slider("最大延迟阶数", 1, 50, 20)
                
                print(f"Computing ACF with max_lag={max_lag}")
                print(f"Output data for ACF: shape={self.output_data.iloc[:,0].values.shape}, non-null={np.sum(~np.isnan(self.output_data.iloc[:,0].values))}")
                acf = self.compute_acf(self.output_data.iloc[:,0].values, max_lag)
                
                print(f"Computing CCF with max_lag={max_lag}")
                print(f"Input data for CCF: shape={self.input_data.iloc[:,0].values.shape}, non-null={np.sum(~np.isnan(self.input_data.iloc[:,0].values))}")
                ccf = self.compute_ccf(self.input_data.iloc[:,0].values, 
                                     self.output_data.iloc[:,0].values, max_lag)
                
                print("Creating correlation plots...")
                fig = make_subplots(rows=2, cols=1,
                                  subplot_titles=('Output Autocorrelation', 'Input-Output Cross-correlation'))
                fig.add_trace(
                    go.Bar(x=list(range(max_lag)), y=acf, name="ACF"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=list(range(max_lag)), y=ccf, name="CCF"),
                    row=2, col=1
                )
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # 添加建议的lag值
                suggested_lags = self.suggest_lags(acf, ccf)
                st.markdown(f"""
                ### 基于相关性分析的建议值
                - 建议的输入延迟阶数: {suggested_lags['input']}
                  - 这个值基于CCF显著相关的范围确定
                - 建议的输出延迟阶数: {suggested_lags['output']}
                  - 这个值基于ACF显著相关的范围确定
                
                > 注意：这些是基于统计分析的建议值，实际使用时可以根据系统特性和模型性能进行调整。
                """)
                
            except Exception as e:
                print("Error in show_data_preview:", str(e))
                st.error(f"Error processing data: {str(e)}")
        print("=== 数据预览完成 ===\n")
    
    def compute_acf(self, data, max_lag=20):
        """计算自相关函数"""
        acf = np.correlate(data - np.mean(data),
                          data - np.mean(data),
                          mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        return acf[:max_lag]
    
    def compute_ccf(self, input_data, output_data, max_lag=20):
        """计算互相关函数"""
        ccf = np.correlate(output_data - np.mean(output_data),
                          input_data - np.mean(input_data),
                          mode='full')
        ccf = ccf / (len(input_data) * np.std(input_data) * np.std(output_data))
        return ccf[len(ccf)//2:len(ccf)//2+max_lag]
    
    def suggest_lags(self, acf, ccf, threshold=0.2):
        """基于相关性分析建议lag值"""
        # 基于显著相关性确定lag值
        input_lag = max(2, len([x for x in ccf if abs(x) > threshold]))
        output_lag = max(2, len([x for x in acf[1:] if abs(x) > threshold]))
        
        return {
            'input': min(input_lag, 10),  # 限制最大lag为10
            'output': min(output_lag, 10)
        } 