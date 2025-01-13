import streamlit as st
import assist.utils as utils
from streamlit.components.v1 import html
from assist.translations import PAGE_TITLES, FILE_UPLOAD_TEXT

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
