import streamlit as st
import os
import pandas as pd
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assist.data_preprocessing import DataPreprocessor
import assist.utils as utils

st.set_page_config(
    page_title="Data Preprocessing - SysIdentPyGUI",
    page_icon="http://sysidentpy.org/overrides/assets/images/favicon.png",
    layout="wide",
)

utils.addlogo()
utils.removemenu()

st.title("数据预处理")

# 初始化预处理器
preprocessor = DataPreprocessor()
preprocessor.load_data()

if "x_data" in st.session_state and "y_data" in st.session_state and \
   st.session_state["x_data"] is not None and st.session_state["y_data"] is not None:
    preprocessor.show_data_preview()
else:
    st.warning("请先在'Load Data'页面上传数据") 