import streamlit as st

st.set_page_config(page_title="BDNLIdentGUI", layout="wide")

# import debugpy; debugpy.breakpoint()  # 调试断点
from assist.translations import PAGE_TITLES
import numpy as np
import pandas as pd
from assist.pagedesign import load_data_section
from assist.data_preprocessing import DataPreprocessor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

import os
import pandas as pd
from assist.assist_dicts import (
    basis_function_list,
    basis_function_parameter_list,
    model_struc_dict,
    model_struc_selec_parameter_list,
    ic_list,
    estimators_list,
    model_type_list,
    los_func_list,
)
import assist.utils as utils
import importlib
from sysidentpy.basis_function import *
from sysidentpy.model_structure_selection import *
from sysidentpy.metrics import __ALL__ as metrics_list
import sysidentpy.metrics as metrics
from sysidentpy.utils.display_results import results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
import pickle as pk
from math import floor
import platform

st.title("动态系统辨识")  # 这个会显示在页面顶部

utils.addlogo()
utils.removemenu()

if platform.system() == "Linux":
    root = os.path.join(os.path.dirname(__file__) + "/assist")
else:
    root = os.path.join(os.path.dirname(__file__) + "\\assist")
path = os.path.join(root, "pagedesign.py")

with open(path, encoding="utf-8") as code:
    c = code.read()
    exec(c, globals())

tabl = ["数据加载", "数据预处理", "模型设置", "模型验证与评估", "保存模型"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(tabl)

with tab1:
    col, esp0, col0 = st.columns([5, 1, 5])

    with col:
        st.file_uploader("输入数据", key="x_data", type=["csv", "xls", "xlsx"], help="拖拽或点击上传CSV或Excel格式文件")
        if st.session_state["x_data"] is not None:
            try:
                file_extension = st.session_state["x_data"].name.split('.')[-1].lower()
                
                if file_extension in ['xls', 'xlsx']:
                    # 读取Excel文件
                    data_x = pd.read_excel(st.session_state["x_data"])
                else:
                    # 尝试不同的编码和分隔符读取CSV文件
                    encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1']
                    separators = ['\t', ',', ';']
                    success = False
                    
                    for encoding in encodings:
                        if success:
                            break
                        try:
                            for sep in separators:
                                try:
                                    st.session_state["x_data"].seek(0)
                                    data_x = pd.read_csv(st.session_state["x_data"], 
                                                       sep=sep, 
                                                       encoding=encoding)
                                    success = True
                                    break
                                except:
                                    continue
                        except:
                            continue
                    
                    if not success:
                        raise Exception("无法读取文件，请检查文件格式和编码")
                
                st.write("输入数据预览：")
                st.write(data_x.head())
                
            except Exception as e:
                st.error(f"读取输入数据时出错: {str(e)}")
                st.info("请确保文件格式正确（CSV格式使用UTF-8编码，或使用Excel格式）")

    with col0:
        st.file_uploader("输出数据", key="y_data", type=["csv", "xls", "xlsx"], help="拖拽或点击上传CSV或Excel格式文件")
        if st.session_state["y_data"] is not None:
            try:
                file_extension = st.session_state["y_data"].name.split('.')[-1].lower()
                
                if file_extension in ['xls', 'xlsx']:
                    # 读取Excel文件
                    data_y = pd.read_excel(st.session_state["y_data"])
                else:
                    # 尝试不同的编码和分隔符读取CSV文件
                    encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1']
                    separators = ['\t', ',', ';']
                    success = False
                    
                    for encoding in encodings:
                        if success:
                            break
                        try:
                            for sep in separators:
                                try:
                                    st.session_state["y_data"].seek(0)
                                    data_y = pd.read_csv(st.session_state["y_data"], 
                                                       sep=sep, 
                                                       encoding=encoding)
                                    success = True
                                    break
                                except:
                                    continue
                        except:
                            continue
                    
                    if not success:
                        raise Exception("无法读取文件，请检查文件格式和编码")
                
                st.write("输出数据预览：")
                st.write(data_y.head())
                
            except Exception as e:
                st.error(f"读取输出数据时出错: {str(e)}")
                st.info("请确保文件格式正确（CSV格式使用UTF-8编码，或使用Excel格式）")

    col1, esp1, esp2 = st.columns([2, 1, 7])
    with col1:
        st.number_input("验证数据比例 (%)", 0.0, 100.0, value=15.0, key="val_perc")
    st.markdown("""---""")
    with st.expander("使用说明"):
        st.write(
            "请上传输入和输出数据的CSV/TXT文件，数据应按列排列（如果有多个输入，请使用制表符分隔）。然后设置用于验证的数据比例。"
        )
        st.write(
            "为获得更好的性能，建议在设置模型后再加载输出数据。需要先加载输入数据以便程序确定输入数量（因此在加载输入数据之前无法设置模型）。"
        )
        st.write(
            "只有在加载了输入和输出数据后才能下载模型。"
        )
        st.write(
            "当更改模型结构选择算法时可能出现问题，按'R'键可以解决。问题原因已知，但在当前代码格式下无法完全修复。"
        )

    if st.session_state["x_data"] != None:
        perc_index = floor(
            data_x.shape[0] - data_x.shape[0] * (st.session_state["val_perc"] / 100)
        )
        x_train, x_valid = (
            data_x[0:perc_index].to_numpy(),
            data_x[perc_index:].to_numpy(),
        )

    if st.session_state["y_data"] != None:
        perc_index = floor(
            data_x.shape[0] - data_x.shape[0] * (st.session_state["val_perc"] / 100)
        )
        y_train, y_valid = (
            data_y[0:perc_index].to_numpy(),
            data_y[perc_index:].to_numpy(),
        )

with tab2:  # 数据预处理 tab
    if 'x_data' in st.session_state and 'y_data' in st.session_state:
        preprocessor = DataPreprocessor()
        preprocessor.load_data()  # 不传递参数，让它从session_state读取数据
        preprocessor.show_data_preview()

with tab3:  # Model Setup tab
    if st.session_state["x_data"] is not None:
        col2, esp3, esp4 = st.columns([2, 1, 1.65])
        with col2:
            # ARX模型设置
            st.markdown("---")  # 分隔线
            st.subheader("ARX模型设置")
            
            # 基函数和阶数设置
            col_arx1, col_arx2, col_arx3 = st.columns(3)
            with col_arx1:
                st.selectbox(
                    "基函数类型", basis_function_list, key="basis_function_key", index=1
                )
            with col_arx2:
                degree = st.number_input("多项式阶数", min_value=1, max_value=5, value=1)
            with col_arx3:
                st.write("基函数参数")
                for i in range(len(basis_function_list)):
                    if st.session_state["basis_function_key"] == basis_function_list[i]:
                        wcont1 = 0
                        key_list = list(basis_function_parameter_list[i])
                        while wcont1 < len(utils.dict_values_to_list(basis_function_parameter_list[i])):
                            st.number_input(
                                key_list[wcont1],
                                value=utils.dict_values_to_list(basis_function_parameter_list[i])[wcont1],
                                key=f"basis_function_parameter_{wcont1}",
                            )
                            wcont1 += 1
            
            # 导入基函数模块
            bf_module = importlib.import_module("sysidentpy.basis_function._basis_function")
            
            # 创建基函数参数字典
            basis_params = {}
            for i in range(len(key_list)):
                param_key = key_list[i]
                param_value = st.session_state[f"basis_function_parameter_{i}"]
                basis_params[param_key] = param_value
            
            # 实例化基函数
            bf = utils.str_to_class(st.session_state["basis_function_key"], bf_module)(**basis_params)

            # 延迟阶数设置
            col_arx4, col_arx5 = st.columns(2)
            with col_arx4:
                na = st.number_input("输出阶数上限 (na)", min_value=1, max_value=10, value=2,
                                   help="将尝试使用1到na之间的所有阶数")
                st.write(f"输出延迟: {list(range(1, na + 1))}")
            with col_arx5:
                nb = st.number_input("输入阶数上限 (nb)", min_value=1, max_value=10, value=2,
                                   help="将尝试使用1到nb之间的所有阶数")
                st.write(f"输入延迟: {list(range(1, nb + 1))}")
            
            # 采样时间设置
            # 检查是否有时间戳列并设置采样时间
            has_timestamp = False
            try:
                if 'data_x' in locals():
                    if 'time' in data_x.columns or 'timestamp' in data_x.columns:
                        time_col = 'time' if 'time' in data_x.columns else 'timestamp'
                        time_data = pd.to_numeric(data_x[time_col], errors='coerce')
                        if not time_data.isna().any():
                            # 计算平均采样时间
                            avg_ts = np.mean(np.diff(time_data))
                            st.write(f"检测到时间序列数据，平均采样时间: {avg_ts:.4f} 秒")
                            has_timestamp = True
                            default_ts = avg_ts
                        else:
                            default_ts = 0.1
                    else:
                        default_ts = 0.1
                else:
                    default_ts = 0.1
            except Exception as e:
                st.warning(f"处理时间戳时出错: {str(e)}")
                default_ts = 0.1
            
            # 允许用户设置采样时间
            Ts = st.number_input(
                "采样时间 (秒)",
                value=float(default_ts),
                min_value=0.0001,
                max_value=1000.0,
                format="%.4f",
                help="如果数据包含时间戳，这里显示检测到的采样时间。您也可以手动调整。"
            )
            
            # 模型结构选择算法
            st.markdown("---")  # 分隔线
            st.subheader("模型结构选择")
            
            st.selectbox(
                "选择算法",
                list(model_struc_dict),
                key="model_struc_select_key",
                index=3,
            )
            
            # 算法参数设置
            for i in range(len(model_struc_dict)):
                if st.session_state["model_struc_select_key"] == list(model_struc_dict)[i]:
                    wcont2 = 0
                    key_list = list(model_struc_selec_parameter_list[i])
                    model_params = {}
                    while wcont2 < len(utils.dict_values_to_list(model_struc_selec_parameter_list[i])):
                        k = "mss_par_" + str(wcont2)
                        param_name = key_list[wcont2]
                        param_value = model_struc_selec_parameter_list[i][param_name]
                        
                        # 跳过maxiter参数
                        if param_name == 'maxiter':
                            wcont2 += 1
                            continue
                            
                        if isinstance(param_value, bool):
                            st.write(utils.adjust_string(param_name))
                            model_params[param_name] = st.checkbox("", key=k, value=param_value)
                        elif isinstance(param_value, int):
                            model_params[param_name] = st.number_input(
                                utils.adjust_string(param_name),
                                key=k,
                                min_value=0,
                                value=param_value
                            )
                        wcont2 += 1

            st.markdown("""---""")

            model_struc_selec_module = importlib.import_module(
                "sysidentpy.model_structure_selection"
                + "."
                + model_struc_dict[st.session_state["model_struc_select_key"]][0]
            )
            model = utils.str_to_class(
                model_struc_dict[st.session_state["model_struc_select_key"]][1],
                model_struc_selec_module,
            )(**model_params)

            # 连续时间模型选择
            st.markdown("---")  # 分隔线
            st.subheader("连续时间模型设置")
            
            model_type = st.selectbox(
                "选择模型结构",
                ["一阶模型 (FOPDT)", "二阶模型 (SOPDT)", "纯延迟模型"]
            )

            if model_type == "一阶模型 (FOPDT)":
                st.write("一阶加延迟模型 (FOPDT)：")
                st.latex(r"G(s) = \frac{K e^{-\theta s}}{\tau s + 1}")
                st.write("其中：K为增益，θ为时滞，τ为时间常数")
            elif model_type == "二阶模型 (SOPDT)":
                st.write("二阶加延迟模型 (SOPDT)：")
                st.latex(r"G(s) = \frac{K e^{-\theta s}}{(\tau_1 s + 1)(\tau_2 s + 1)}")
                st.write("其中：K为增益，θ为时滞，τ₁和τ₂为时间常数")
            else:
                st.write("纯延迟模型：")
                st.latex(r"G(s) = K e^{-\theta s}")
                st.write("其中：K为增益，θ为时滞")

            # 参数估计按钮
            if st.button("开始ARX模型估计"):
                if 'data_x' in locals() and 'data_y' in locals():
                    try:
                        st.write("### ARX模型参数估计结果")
                        
                        # 使用ARX模型，设置ylag和xlag为所有可能的延迟组合
                        model.ylag = list(range(1, na + 1))  # [1, 2, ..., na]
                        model.xlag = list(range(1, nb + 1))  # [1, 2, ..., nb]
                        
                        # 拟合模型
                        model.fit(X=x_train, y=y_train)
                        
                        # 保存模型到session_state
                        st.session_state['fitted_model'] = model
                        
                        # 生成预测
                        if isinstance(model, MetaMSS):  # MetaMSS有不同的方法
                            yhat = model.predict(X=x_valid, y=y_valid)
                        else:
                            yhat = model.predict(X=x_valid, y=y_valid)
                        
                        # 保存预测结果
                        st.session_state['yhat'] = yhat
                        st.session_state['y_valid'] = y_valid
                        st.session_state['x_valid'] = x_valid
                        
                        # 显示预测结果
                        plt.figure(figsize=(10, 6))
                        plt.plot(y_valid, label='实际值', color='blue')
                        plt.plot(yhat, label='预测值', color='red', linestyle='--')
                        plt.title("ARX模型预测结果")
                        plt.xlabel('样本')
                        plt.ylabel('值')
                        plt.legend()
                        st.pyplot(plt)
                        
                    except Exception as e:
                        st.error(f"处理数据时出错: {str(e)}")
                        st.info("请确保数据格式正确，并且已经上传了输入和输出数据")
                else:
                    st.warning("请先在数据加载页面上传输入和输出数据")

            # 连续时间模型参数估计按钮
            if st.button("开始连续时间模型估计"):
                if 'data_x' in locals() and 'data_y' in locals():
                    try:
                        st.write("### 连续时间模型参数估计结果")
                        
                        # 获取数据
                        try:
                            if has_timestamp and len(data_x.columns) > 1:
                                # 如果有时间戳且有多列，使用除时间戳外的第一列
                                x = data_x.iloc[:, [i for i in range(len(data_x.columns)) if data_x.columns[i] not in ['time', 'timestamp']][0]].values
                            else:
                                # 否则使用第一列
                                x = data_x.iloc[:, 0].values
                        except Exception as e:
                            st.error(f"选择输入数据列时出错: {str(e)}")
                            x = data_x.iloc[:, 0].values
                            
                        y = data_y.iloc[:, 0].values
                        
                        if model_type == "一阶模型 (FOPDT)":
                            # 使用ARX模型，设置ylag和xlag为所有可能的延迟组合
                            model.ylag = list(range(1, na + 1))  # [1, 2, ..., na]
                            model.xlag = list(range(1, nb + 1))  # [1, 2, ..., nb]
                            
                            # 拟合模型
                            model.fit(X=x_train, y=y_train)
                            
                            # 获取ARX参数
                            arx_params = {
                                'a': model.theta[:na],  # 输出系数
                                'b': model.theta[na:na+nb],  # 输入系数
                            }
                            
                            # 转换为FOPDT参数
                            fopdt_params = utils.convert_arx_to_continuous(arx_params, Ts)
                            st.session_state['model_params'] = fopdt_params
                            
                            st.write(f"估计的时滞 θ = {fopdt_params['theta']:.4f} 秒")
                            st.write(f"增益 K = {fopdt_params['K']:.4f}")
                            st.write(f"时间常数 τ = {fopdt_params['tau']:.4f} 秒")
                            
                            # 生成预测值
                            y_pred = np.zeros_like(y)
                            delay_samples = int(fopdt_params['theta'] / Ts)
                            for i in range(len(y)):
                                if i+delay_samples < len(x):
                                    # 使用FOPDT模型计算预测值
                                    t = i * Ts
                                    y_pred[i] = fopdt_params['K'] * x[i] * (1 - np.exp(-t/fopdt_params['tau']))
                            
                            # 保存预测结果
                            st.session_state['y_pred_ct'] = y_pred
                            
                        elif model_type == "二阶模型 (SOPDT)":
                            # 使用ARX模型，设置ylag和xlag为所有可能的延迟组合
                            model.ylag = list(range(1, na + 1))  # [1, 2, ..., na]
                            model.xlag = list(range(1, nb + 1))  # [1, 2, ..., nb]
                            
                            # 拟合模型
                            model.fit(X=x_train, y=y_train)
                            
                            # 获取ARX参数
                            arx_params = {
                                'a': model.theta[:na],  # 输出系数
                                'b': model.theta[na:na+nb],  # 输入系数
                            }
                            
                            # 转换为SOPDT参数
                            sopdt_params = utils.convert_arx_to_continuous(arx_params, Ts)
                            st.session_state['model_params'] = sopdt_params
                            
                            st.write(f"估计的时滞 θ = {sopdt_params['theta']:.4f} 秒")
                            st.write(f"增益 K = {sopdt_params['K']:.4f}")
                            st.write(f"时间常数 τ₁ = {sopdt_params['tau1']:.4f} 秒")
                            st.write(f"时间常数 τ₂ = {sopdt_params['tau2']:.4f} 秒")
                            
                            # 生成预测值
                            y_pred = np.zeros_like(y)
                            delay_samples = int(sopdt_params['theta'] / Ts)
                            for i in range(len(y)):
                                if i+delay_samples < len(x):
                                    # 使用SOPDT模型计算预测值
                                    t = i * Ts
                                    y_pred[i] = sopdt_params['K'] * x[i] * (
                                        1 - (sopdt_params['tau1'] * np.exp(-t/sopdt_params['tau1']) - 
                                             sopdt_params['tau2'] * np.exp(-t/sopdt_params['tau2'])) / 
                                        (sopdt_params['tau1'] - sopdt_params['tau2'])
                                    )
                            
                            # 保存预测结果
                            st.session_state['y_pred_ct'] = y_pred
                            
                        else:  # 纯延迟模型
                            # 估计时滞
                            corr = np.correlate(y - np.mean(y), x - np.mean(x), mode='full')
                            delay = len(corr)//2 - np.argmax(corr)
                            theta = delay * Ts
                            
                            # 估计增益K
                            K = np.std(y) / np.std(x)
                            
                            # 保存模型参数
                            st.session_state['model_params'] = {
                                'theta': theta,
                                'K': K
                            }
                            
                            st.write(f"估计的时滞 θ = {theta:.4f} 秒")
                            st.write(f"增益 K = {K:.4f}")
                            
                            # 生成预测值
                            y_pred = np.zeros_like(y)
                            for i in range(len(y)):
                                if i+delay < len(x):
                                    y_pred[i] = K * x[i]
                            
                            # 保存预测结果
                            st.session_state['y_pred_ct'] = y_pred

                        # 计算评估指标
                        valid_idx = ~np.isnan(y_pred)
                        r2 = r2_score(y[valid_idx], y_pred[valid_idx])
                        rmse = np.sqrt(mean_squared_error(y[valid_idx], y_pred[valid_idx]))
                        st.write("### 模型评估")
                        st.write(f"R² 分数: {r2:.4f}")
                        st.write(f"RMSE: {rmse:.4f}")

                        # 绘制结果对比图
                        plt.figure(figsize=(10, 6))
                        plt.plot(y, label='实际值', color='blue')
                        plt.plot(y_pred, label='预测值', color='red', linestyle='--')
                        plt.title(f"{model_type}预测结果对比")
                        plt.xlabel('样本')
                        plt.ylabel('值')
                        plt.legend()
                        st.pyplot(plt)
                        
                    except Exception as e:
                        st.error(f"处理数据时出错: {str(e)}")
                        st.info("请确保数据格式正确，并且已经上传了输入和输出数据")
                else:
                    st.warning("请先在数据加载页面上传输入和输出数据")

with tab4:  # Model Validation tab
    if (
        st.session_state["y_data"] != None and st.session_state["x_data"] != None
    ):
        # 创建两个标签页
        val_tab1, val_tab2 = st.tabs(["ARX模型验证", "连续时间模型验证"])
        
        with val_tab1:  # ARX模型验证
            # 检查模型是否已经拟合
            if 'fitted_model' not in st.session_state:
                st.warning("请先在模型设置页面完成ARX模型参数估计")
            else:
                model = st.session_state['fitted_model']
                st.write("模型回归器")
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
                st.dataframe(r)

                # 检查是否有预测结果
                if 'yhat' not in st.session_state:
                    st.warning("请先在模型设置页面完成ARX模型预测")
                else:
                    yhat = st.session_state['yhat']
                    y_valid = st.session_state['y_valid']
                    x_valid = st.session_state['x_valid']
                    
                    ee = compute_residues_autocorrelation(y_valid, yhat)
                    if x_valid.shape[1] == 1:
                        x1e = compute_cross_correlation(y_valid, yhat, x_valid)
                    else:
                        x1e = compute_cross_correlation(y_valid, yhat, x_valid[:, 0])
                    
                    with st.expander("结果图"):
                        if "free_run" not in st.session_state:
                            st.session_state["free_run"] = True
                            
                        if st.session_state["free_run"]:
                            st.image(utils.plot_results(y=y_valid, yhat=yhat, n=1000))
                        else:
                            st.image(
                                utils.plot_results(
                                    y=y_valid,
                                    yhat=yhat,
                                    n=1000,
                                    title=str(st.session_state.get("steps_ahead", ""))
                                    + " 步预测仿真",
                                )
                            )
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
                                    getattr(metrics, metrics_list[index])(y_valid, yhat)
                                )
                        metrics_df["指标名称"] = metrics_namelist
                        metrics_df["数值"] = metrics_vallist
                        st.dataframe(pd.DataFrame(metrics_df).style.format({"数值": "{:f}"}))
        
        with val_tab2:  # 连续时间模型验证
            if 'model_params' not in st.session_state:
                st.warning("请先在模型设置页面完成连续时间模型参数估计")
            elif 'y_pred_ct' not in st.session_state:
                st.warning("请先在模型设置页面完成连续时间模型预测")
            else:
                st.write("### 连续时间模型参数")
                params = st.session_state['model_params']
                for key, value in params.items():
                    st.write(f"{key} = {value:.4f}")
                
                y_pred = st.session_state['y_pred_ct']
                valid_idx = ~np.isnan(y_pred)
                r2 = r2_score(y[valid_idx], y_pred[valid_idx])
                rmse = np.sqrt(mean_squared_error(y[valid_idx], y_pred[valid_idx]))
                
                st.write("### 模型评估")
                st.write(f"R² 分数: {r2:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                
                # 绘制结果对比图
                plt.figure(figsize=(10, 6))
                plt.plot(y, label='实际值', color='blue')
                plt.plot(y_pred, label='预测值', color='red', linestyle='--')
                plt.title(f"{model_type}预测结果对比")
                plt.xlabel('样本')
                plt.ylabel('值')
                plt.legend()
                st.pyplot(plt)

with tab5:  # Save Model tab
    if st.session_state["y_data"] != None and st.session_state["x_data"] != None:
        st.download_button(
            "下载模型",
            data=pk.dumps(model),
            file_name="my_model.syspy"
        )
