import streamlit as st
from assist.translations import PAGE_TITLES

st.set_page_config(
    page_title=PAGE_TITLES["Simulate_a_predefined_model"],  # 使用翻译字典中的标题
    page_icon="🔬",
    layout="wide",
)

from sysidentpy.simulation import SimulateNARMAX
from sysidentpy.basis_function._basis_function import Polynomial
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
import assist.utils as utils
import pandas as pd
import numpy as np
import platform

if platform.system() == "Linux":
    root = os.path.join(os.path.dirname(__file__).replace("/pages", "") + "/assist")
else:
    root = os.path.join(os.path.dirname(__file__).replace("\pages", "") + "\\assist")
path = os.path.join(root, "pagedesign.py")

with open(path, encoding="utf-8") as code:
    c = code.read()
    exec(c, globals())

st.title("模型仿真")  # 这个会显示在页面顶部

utils.addlogo()
utils.removemenu()

def add_regressors(regr_index_key, regr_list_key):
    regkeys = utils.sorter(
        [regk for regk in list(st.session_state) if regr_index_key in regk]
    )
    nested_regressors = [st.session_state[regk] for regk in regkeys]
    st.session_state[regr_list_key].append(nested_regressors)


def add_thetas(thetas_index_key, thetas_list_key):
    thetaskeys = utils.sorter(
        [thetask for thetask in list(st.session_state) if thetas_index_key in thetask]
    )
    nested_thetas = [st.session_state[thetask] for thetask in thetaskeys]
    st.session_state[thetas_list_key].append(nested_thetas)


col, esp0, col0 = st.columns([5, 1, 5])

with col:
    st.file_uploader(
        "测试输入数据", key="test_x_data", help="拖拽或点击上传CSV格式文件"
    )
    if st.session_state["test_x_data"] != None:
        test_x = pd.read_csv(
            st.session_state["test_x_data"], sep="\t", header=None
        ).to_numpy()

with col0:
    st.file_uploader(
        "测试输出数据", key="test_y_data", help="拖拽或点击上传CSV格式文件"
    )
    if st.session_state["test_y_data"] != None:
        test_y = pd.read_csv(
            st.session_state["test_y_data"], sep="\t", header=None
        ).to_numpy()

if st.session_state["test_x_data"] != None:
    st.markdown("""---""")
    st.write("模型配置")
    col1, esp1, esp2 = st.columns([2, 1, 7])
    with col1:
        st.number_input("非线性度", key="nl_degree", min_value=1)

    st.markdown("""---""")
    st.write("回归器列表")

    if "regr_list" not in st.session_state:
        st.session_state["regr_list"] = list()

    col2, esp3, esp4 = st.columns([2, 1, 7])
    with col2:
        st.number_input("回归器数量", key="n_regr", min_value=1)

    col3, esp5, esp6 = st.columns([2, 1, 7])
    with col3:
        for i in range(st.session_state["n_regr"]):
            st.number_input(
                "回归器 " + str(i + 1), key="regr_" + str(i + 1), min_value=1
            )

        st.button("添加回归器组", on_click=add_regressors, args=("regr_", "regr_list"))

    st.write("当前回归器列表:")
    st.write(st.session_state["regr_list"])

    if len(st.session_state["regr_list"]) > 0:
        if st.button("清空回归器列表"):
            st.session_state["regr_list"] = list()

    st.markdown("""---""")
    st.write("参数列表")

    if "thetas_list" not in st.session_state:
        st.session_state["thetas_list"] = list()

    col4, esp7, esp8 = st.columns([2, 1, 7])
    with col4:
        st.number_input("参数数量", key="n_thetas", min_value=1)

    col5, esp9, esp10 = st.columns([2, 1, 7])
    with col5:
        for i in range(st.session_state["n_thetas"]):
            st.number_input(
                "参数 " + str(i + 1), key="theta_" + str(i + 1)
            )

        st.button("添加参数组", on_click=add_thetas, args=("theta_", "thetas_list"))

    st.write("当前参数列表:")
    st.write(st.session_state["thetas_list"])

    if len(st.session_state["thetas_list"]) > 0:
        if st.button("清空参数列表"):
            st.session_state["thetas_list"] = list()

    st.markdown("""---""")
    st.write("仿真选项")

    if "steps_ahead" not in st.session_state:
        st.session_state["steps_ahead"] = None
    if "forecast_horizon" not in st.session_state:
        st.session_state["forecast_horizon"] = None

    st.write("自由运行仿真")
    if st.checkbox("", value=True, key="free_run") is False:
        st.number_input("预测步数", key="steps_ahead", min_value=1)
        st.number_input("预测范围", key="forecast_horizon", min_value=1)

    if st.button("运行模型仿真"):
        if st.session_state["test_y_data"] != None:
            basis_function = Polynomial(degree=st.session_state["nl_degree"])
            model = SimulateNARMAX(
                basis_function=basis_function,
                xlag=[[1], [1]],
                ylag=[[1], [1]],
                model_type="NARMAX",
            )
            model.fit(X=test_x, y=test_y)
            model.basis_function.model_type = "NARMAX"
            model.basis_function.build_output_matrix(test_x, test_y)
            model.basis_function.n_inputs = test_x.shape[1]
            model.basis_function.max_lag = max(
                max(model.xlag[0]), max(model.ylag[0])
            )
            model.basis_function.n_terms = len(st.session_state["regr_list"])
            model.final_model = st.session_state["regr_list"]
            model.theta = st.session_state["thetas_list"]
            yhat = model.predict(
                X=test_x,
                y=test_y,
                steps_ahead=st.session_state["steps_ahead"],
                forecast_horizon=st.session_state["forecast_horizon"],
            )

            st.markdown("""---""")
            st.write("模型方程")
            r = pd.DataFrame(
                results(
                    model.final_model,
                    model.theta,
                    model.err,
                    model.n_terms,
                    err_precision=8,
                    dtype="sci",
                ),
                columns=["回归项", "参数", "ERR"],
            )
            st.write(r)

            metrics_df = dict()
            metrics_namelist = list()
            metrics_vallist = list()
            with st.expander("评估指标"):
                for index in range(len(metrics_list)):
                    if metrics_list[index] == "forecast_error":
                        pass
                    else:
                        metrics_namelist.append(
                            utils.get_acronym(utils.adjust_string(metrics_list[index]))
                        )
                        metrics_vallist.append(
                            getattr(metrics, metrics_list[index])(test_y, yhat)
                        )
                metrics_df["指标名称"] = metrics_namelist
                metrics_df["数值"] = metrics_vallist
                st.dataframe(pd.DataFrame(metrics_df).style.format({"数值": "{:f}"}))

            with st.expander("结果图"):
                if st.session_state["free_run"] == True:
                    st.image(utils.plot_results(y=test_y, yhat=yhat, n=1000))
                else:
                    st.image(
                        utils.plot_results(
                            y=test_y,
                            yhat=yhat,
                            n=1000,
                            title=str(st.session_state["steps_ahead"])
                            + " 步预测仿真",
                        )
                    )

            ee = compute_residues_autocorrelation(test_y, yhat)
            if test_x.shape[1] == 1:
                x1e = compute_cross_correlation(test_y, yhat, test_x)
            else:
                x1e = compute_cross_correlation(test_y, yhat, test_x[:, 0])

            with st.expander("残差图"):
                st.image(
                    utils.plot_residues_correlation(
                        data=ee, title="残差", ylabel="$e^2$"
                    )
                )
                st.image(
                    utils.plot_residues_correlation(
                        data=x1e, title="残差", ylabel="$x_1e$", second_fig=True
                    )
                )
