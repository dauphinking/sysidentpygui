import streamlit as st
from assist.translations import PAGE_TITLES

st.set_page_config(
    page_title=PAGE_TITLES["Data_Preprocessing"],
    page_icon="📊",
    layout="wide",
)

import os
import pandas as pd
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assist.data_preprocessing import DataPreprocessor
from assist.translations import PAGE_TITLES
import assist.utils as utils

st.title("数据预处理")

utils.addlogo()
utils.removemenu()

# 初始化预处理器
preprocessor = DataPreprocessor()
preprocessor.load_data()

if "x_data" in st.session_state and "y_data" in st.session_state and \
   st.session_state["x_data"] is not None and st.session_state["y_data"] is not None:
    preprocessor.show_data_preview()
else:
    st.warning("请先在'数据加载'页面上传数据") 