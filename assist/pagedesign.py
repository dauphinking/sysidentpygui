import streamlit as st
import assist.utils as utils
from streamlit.components.v1 import html
from assist.translations import PAGE_TITLES, FILE_UPLOAD_TEXT
import pandas as pd

# 修改文件上传提示文本
st.markdown(
    """
    <style>
        .uploadedFile {
            display: none;
        }
        .stFileUploader > section > input::file-selector-button {
            content: "%s";
        }
        .stFileUploader > section > div[data-testid="stMarkdownContainer"] > p {
            font-size: 14px;
            color: rgb(49, 51, 63);
        }
        .stFileUploader > section > div[data-testid="stMarkdownContainer"] > p:first-child::before {
            content: "%s\\A";
            white-space: pre;
        }
        .stFileUploader > section > div[data-testid="stMarkdownContainer"] > p:last-child::before {
            content: "%s";
        }
        /* 修改侧边栏页面名称显示 */
        section[data-testid="stSidebar"] .css-17lntkn {
            display: none;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            padding: 0.5rem 1rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background-color: rgba(151, 166, 195, 0.15);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a p {
            font-size: 1rem;
        }
    </style>
    """ % (FILE_UPLOAD_TEXT["browse_files"], FILE_UPLOAD_TEXT["drag_drop"], FILE_UPLOAD_TEXT["file_limit"]),
    unsafe_allow_html=True
)

def load_data_section():
    """加载数据部分"""
    st.header("数据加载")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 上传输入数据
        x_data = st.file_uploader("上传输入数据 (CSV格式)", type=['csv'], key='x_data_uploader')
        if x_data is not None:
            st.session_state['x_data'] = x_data
            
    with col2:
        # 上传输出数据
        y_data = st.file_uploader("上传输出数据 (CSV格式)", type=['csv'], key='y_data_uploader')
        if y_data is not None:
            st.session_state['y_data'] = y_data
            
    # 如果两个文件都上传了，保存到session state
    if x_data is not None and y_data is not None:
        try:
            # 读取数据
            x_df = pd.read_csv(x_data, header=None)
            y_df = pd.read_csv(y_data, header=None)
            
            # 保存DataFrame到session state
            st.session_state['x_data_df'] = x_df
            st.session_state['y_data_df'] = y_df
            st.session_state['data_loaded'] = True
            
            # 显示成功消息
            st.success("数据加载成功！")
        except Exception as e:
            st.error(f"数据加载失败，错误信息：{e}")
