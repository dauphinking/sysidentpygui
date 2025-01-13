import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from io import StringIO

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
                # 读取输入数据
                data = st.session_state["x_data"].getvalue().decode('utf-8').splitlines()
                data = [line.strip() for line in data if line.strip()]  # 去除空行和空白
                print(f"First line of input data: {data[0]}")
                
                if '\t' in data[0]:  # 多列数据，用制表符分隔
                    print("Detected multi-column data")
                    values = []
                    for line in data:
                        try:
                            # 使用pandas直接读取制表符分隔的字符串
                            row = pd.read_csv(StringIO(line), sep='\t', header=None).iloc[0].values
                            values.append(row)
                        except Exception as e:
                            print(f"Error parsing line: {line}")
                            raise e
                    self.input_data = pd.DataFrame(values)
                    print("Loaded multi-column input data")
                else:  # 单列数据
                    print("Detected single-column data")
                    values = []
                    for line in data:
                        try:
                            values.append(float(line))
                        except ValueError as e:
                            print(f"Error parsing line: {line}")
                            raise e
                    self.input_data = pd.DataFrame(values, columns=['value'])
                    print("Loaded single-column input data")
                print("Input data shape:", self.input_data.shape)
                print("Input data head:\n", self.input_data.head())
            except Exception as e:
                print("Error loading input data:", str(e))
                self.input_data = None
                
        if "y_data" in st.session_state and st.session_state["y_data"] is not None:
            try:
                # 读取输出数据
                data = st.session_state["y_data"].getvalue().decode('utf-8').splitlines()
                data = [line.strip() for line in data if line.strip()]  # 去除空行和空白
                values = []
                for line in data:
                    try:
                        values.append(float(line))
                    except ValueError as e:
                        print(f"Error parsing line: {line}")
                        raise e
                self.output_data = pd.DataFrame(values, columns=['value'])
                print("Output data loaded successfully")
                print("Output data shape:", self.output_data.shape)
                print("Output data head:\n", self.output_data.head())
            except Exception as e:
                print("Error loading output data:", str(e))
                self.output_data = None
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
                n_inputs = self.input_data.shape[1]
                fig = make_subplots(rows=n_inputs+1, cols=1,
                                  subplot_titles=['Input Time Series ' + str(i+1) for i in range(n_inputs)] + ['Output Time Series'])
                
                # Plot each input time series
                for i in range(n_inputs):
                    fig.add_trace(
                        go.Scatter(y=self.input_data.iloc[:,i], name=f"Input {i+1}"),
                        row=i+1, col=1
                    )
                
                # Plot output time series
                fig.add_trace(
                    go.Scatter(y=self.output_data.iloc[:,0], name="Output"),
                    row=n_inputs+1, col=1
                )
                
                fig.update_layout(height=300*(n_inputs+1), showlegend=True)
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
                - CCF的横坐标表示延迟：
                  - 正值：输入领先于输出的延迟（通常关注这部分）
                  - 负值：输入滞后于输出的延迟（通常不考虑）
                  - 0：无延迟
                - CCF的峰值表明输入对输出的最强影响出现的延迟
                - 建议选择从0到CCF绝对值最大的正延迟位置作为输入延迟阶数
                
                #### 选择建议
                - 较大的延迟会增加模型复杂度
                - 建议从较小的延迟开始（如2-3），根据模型性能逐步调整
                - 如果ACF/CCF在较大延迟处仍显著，可以考虑增加延迟阶数
                """)
                
                max_lag = st.slider("最大延迟阶数", 1, 50, 20)
                
                print(f"Computing ACF with max_lag={max_lag}")
                print(f"Output data for ACF: shape={self.output_data.iloc[:,0].values.shape}, non-null={np.sum(~np.isnan(self.output_data.iloc[:,0].values))}")
                acf = self.compute_acf(self.output_data.iloc[:,0].values, max_lag)
                
                # 为每个输入变量计算CCF
                n_inputs = self.input_data.shape[1]
                ccfs = []
                max_ccf_lags = []  # Store the lag with the maximum absolute CCF value
                max_ccf_values = []  # Store the maximum CCF values
                for i in range(n_inputs):
                    print(f"Computing CCF for input {i+1} with max_lag={max_lag}")
                    ccf = self.compute_ccf(self.input_data.iloc[:,i].values, self.output_data.iloc[:,0].values, max_lag)
                    ccfs.append(ccf)
                    # Find the maximum absolute CCF value and its corresponding lag
                    abs_ccf = np.abs(ccf)
                    max_abs_idx = np.argmax(abs_ccf)
                    max_ccf_lag = max_abs_idx - max_lag  # Convert to actual lag value
                    max_ccf_value = ccf[max_abs_idx]  # Retain the original sign
                    max_ccf_lags.append(max_ccf_lag)
                    max_ccf_values.append(max_ccf_value)
                    print(f"Input {i+1} - Max CCF: {max_ccf_value:.3f} at lag {max_ccf_lag}")

                print("Creating correlation plots...")
                fig = make_subplots(rows=n_inputs+1, cols=1,
                                  subplot_titles=['Output ACF'] + [f'Input {i+1} CCF' for i in range(n_inputs)])
                
                # Plot ACF
                fig.add_trace(
                    go.Bar(x=list(range(max_lag+1)), y=acf, name="ACF"),
                    row=1, col=1
                )
                
                # Plot CCFs
                for i, ccf in enumerate(ccfs):
                    trace = go.Bar(x=list(range(-max_lag, max_lag+1)), y=ccf, name=f"CCF (Input {i+1})")
                    fig.add_trace(trace, row=i+2, col=1)
                    # Add vertical lines to mark the maximum CCF position
                    fig.add_vline(x=max_ccf_lags[i], line_dash="dash", line_color="red",
                                  annotation_text=f"Max Correlation: {max_ccf_values[i]:.3f}", row=i+2, col=1)
                    # Add vertical line to mark zero lag position
                    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                                  annotation_text="Zero Lag", row=i+2, col=1)
                
                fig.update_layout(height=300*(n_inputs+1), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # 根据相关性分析给出建议
                st.subheader("延迟阶数建议")
                
                # ACF分析建议
                significant_acf_lags = np.where(np.abs(acf) > 0.2)[0]
                suggested_output_lag = min(max(significant_acf_lags) if len(significant_acf_lags) > 0 else 2, 5)
                st.write(f"基于ACF分析，建议输出延迟阶数: {suggested_output_lag}")
                
                # CCF分析建议
                for i in range(n_inputs):
                    st.write(f"输入{i+1}的CCF分析：")
                    st.write(f"- 最大相关性 {max_ccf_values[i]:.3f} 出现在延迟 {max_ccf_lags[i]} 处")
                    if max_ccf_lags[i] > 0:
                        st.write(f"- 输入领先于输出 {max_ccf_lags[i]} 个时间步")
                        st.write(f"- 建议的输入延迟阶数: {min(max_ccf_lags[i] + 1, 5)}")
                    elif max_ccf_lags[i] < 0:
                        st.write(f"- 输出领先于输入 {abs(max_ccf_lags[i])} 个时间步")
                        st.write("- 建议从较小的延迟阶数(2-3)开始尝试，因为输入滞后于输出")
                    else:
                        st.write("- 最强相关性在零延迟处")
                        st.write("- 建议使用较小的延迟阶数(1-2)")
                    
                    if max_ccf_values[i] > 0:
                        st.write("- 呈正相关关系")
                    else:
                        st.write("- 呈负相关关系")
                
                print("=== 数据预览完成 ===")
                
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