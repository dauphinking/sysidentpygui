import streamlit as st
from assist.translations import PAGE_TITLES

st.set_page_config(
    page_title=PAGE_TITLES["Load_your_model"],  # 使用翻译字典中的标题
    page_icon="📂",
    layout="wide",
)

from sysidentpy.utils.display_results import results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sysidentpy.metrics import __ALL__ as metrics_list
import sysidentpy.metrics as metrics
import os
import sys
import assist.utils as utils

sys.path.insert(1, os.path.dirname(__file__).replace("\pages", ""))
import pandas as pd
import pickle as pk
import platform

if platform.system() == "Linux":
    root = os.path.join(os.path.dirname(__file__).replace("/pages", "") + "/assist")
else:
    root = os.path.join(os.path.dirname(__file__).replace("\pages", "") + "\\assist")
path = os.path.join(root, "pagedesign.py")

with open(path, encoding="utf-8") as code:
    c = code.read()
    exec(c, globals())

st.title("加载模型")  # 这个会显示在页面顶部

utils.addlogo()
utils.removemenu()

col, esp0, col0 = st.columns([5, 1, 5])

with col:
    st.file_uploader(
        "验证输入数据", key="vx_data", help="拖拽或点击上传CSV格式文件"
    )
    if st.session_state["vx_data"] != None:
        x_valid = pd.read_csv(
            st.session_state["vx_data"], sep="\t", header=None
        ).to_numpy()

with col0:
    st.file_uploader(
        "验证输出数据", key="vy_data", help="拖拽或点击上传CSV格式文件"
    )
    if st.session_state["vy_data"] != None:
        y_valid = pd.read_csv(
            st.session_state["vy_data"], sep="\t", header=None
        ).to_numpy()

with col:
    st.file_uploader("加载模型文件", key="model_file", help="拖拽或点击上传模型文件")

if st.session_state["model_file"] != None:
    loaded_model = pk.load(st.session_state["model_file"])

st.markdown("---")

if (
    st.session_state["model_file"] != None
    and st.session_state["vx_data"] != None
    and st.session_state["vy_data"] != None
):
    yhat_loaded = loaded_model.predict(X=x_valid, y=y_valid)

    r_loaded = pd.DataFrame(
        results(
            loaded_model.final_model,
            loaded_model.theta,
            loaded_model.err,
            loaded_model.n_terms,
            err_precision=8,
            dtype="sci",
        ),
        columns=["回归项", "参数", "ERR"],
    )
    st.subheader("已加载模型 \n")
    st.write(r_loaded)

    metrics_df = dict()
    metrics_namelist = list()
    metrics_vallist = list()
    with st.expander("评估指标"):
        for index in range(len(metrics_list)):
            if (
                metrics_list[index] == "r2_score"
                or metrics_list[index] == "forecast_error"
            ):
                pass
            else:
                metrics_namelist.append(
                    utils.get_acronym(utils.adjust_string(metrics_list[index]))
                )
                metrics_vallist.append(
                    getattr(metrics, metrics_list[index])(y_valid, yhat_loaded)
                )
        metrics_df["指标名称"] = metrics_namelist
        metrics_df["数值"] = metrics_vallist
        st.dataframe(pd.DataFrame(metrics_df).style.format({"数值": "{:f}"}))

    with st.expander("结果图"):
        st.image(utils.plot_results(y=y_valid, yhat=yhat_loaded, n=1000))

    ee = compute_residues_autocorrelation(y_valid, yhat_loaded)
    if x_valid.shape[1] == 1:
        x1e = compute_cross_correlation(y_valid, yhat_loaded, x_valid)
    else:
        x1e = compute_cross_correlation(y_valid, yhat_loaded, x_valid[:, 0])

    with st.expander("残差图"):
        st.image(
            utils.plot_residues_correlation(data=ee, title="残差", ylabel="$e^2$")
        )
        st.image(
            utils.plot_residues_correlation(
                data=x1e, title="残差", ylabel="$x_1e$", second_fig=True
            )
        )
