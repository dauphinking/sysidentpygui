import streamlit as st
import sys
import re
import matplotlib.pyplot as plt
from PIL import Image
from sysidentpy.utils.display_results import results
import pandas as pd
import inspect
import matplotlib.font_manager as fm

# 设置中文字体
def set_chinese_font():
    # 尝试多个常见的中文字体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    font_found = False
    
    for font_name in chinese_fonts:
        try:
            fm.findfont(font_name)
            plt.rcParams['font.family'] = font_name
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        print("Warning: No suitable Chinese font found. Using default font.")
    
    # 确保负号正确显示
    plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
try:
    # 尝试使用微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
except:
    try:
        # 尝试使用其他中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    except:
        print("Warning: Could not find Chinese fonts. Some characters may not display correctly.")

# 设置负号的正确显示
plt.rcParams['axes.unicode_minus'] = False


def addlogo():
    st.markdown(
        """<img src="https://www.biditech.cn/_next/image?url=%2Fimages%2Fbiditech-logo.png&w=256&q=75" alt="logo" class="center"> """,
        unsafe_allow_html=True,
    )  # adiciona a logo
    st.markdown(
        """ <style> 
    .center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 35%;
    }
    </style>""",
        unsafe_allow_html=True,
    )


def removemenu(op=True):
    if op == True:
        st.markdown(
            """ <style> 
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>""",
            unsafe_allow_html=True,
        )


def get_estimators(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != "_"]


def get_default_args(func):
    """
    Returns a dictionary containing the default arguments of the given function.
    """
    sig = inspect.signature(func)
    return {arg.name: arg.default for arg in sig.parameters.values()}


def get_model_struc_dict(
    module, prefix
):  # creates a dictionary that the app uses for the method name and objects instantiating
    _model_struc_dict = dict()
    _cls_names = [
        cls_name
        for cls_name, _ in inspect.getmembers(
            module, lambda member: inspect.isclass(member)
        )
    ]
    _cls_modules_paths = [
        getattr(module, classname).__module__ for classname in _cls_names
    ]
    _module_names = [modpath.replace(prefix, "") for modpath in _cls_modules_paths]

    for i in range(len(_cls_names)):
        if _cls_names[i] == "FROLS":
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + " Compact)"
            ] = [_module_names[i], _cls_names[i]]
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + " Complete)"
            ] = [_module_names[i], _cls_names[i]]
        elif _cls_names[i] == "MetaMSS":
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + " Compact)"
            ] = [_module_names[i], _cls_names[i]]
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + " Complete)"
            ] = [_module_names[i], _cls_names[i]]
        elif _cls_names[i] == "ER":
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + " Compact)"
            ] = [_module_names[i], _cls_names[i]]
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + " Complete)"
            ] = [_module_names[i], _cls_names[i]]
        else:
            _model_struc_dict[
                adjust_string(_module_names[i]) + " (" + _cls_names[i] + ")"
            ] = [_module_names[i], _cls_names[i]]

    return _model_struc_dict


def get_model_struc_selec_parameter_list(module):
    _cls_names = [
        cls_name
        for cls_name, _ in inspect.getmembers(
            module, lambda member: inspect.isclass(member)
        )
    ]
    _model_struc_selec_parameter_list = list()
    for _cls in _cls_names:
        if _cls == "FROLS":
            _full_args = get_default_args(getattr(module, _cls))
            _compact_form_keys = [
                "order_selection",
                "n_terms",
                "n_info_values",
                "extended_least_squares",
                "ylag",
                "xlag",
                "info_criteria",
                "estimator",
                "model_type",
                "basis_function",
            ]
            _compact_args = dict(((key, _full_args[key]) for key in _compact_form_keys))
            _model_struc_selec_parameter_list.append(_compact_args)
            _model_struc_selec_parameter_list.append(_full_args)
        elif _cls == "MetaMSS":
            _full_args = get_default_args(getattr(module, _cls))
            _compact_form_keys = [
                "maxiter",
                "k_agents_percent",
                "norm",
                "n_agents",
                "xlag",
                "ylag",
                "estimator",
                "estimate_parameter",
                "loss_func",
                "model_type",
                "basis_function",
            ]
            _compact_args = dict(((key, _full_args[key]) for key in _compact_form_keys))
            _model_struc_selec_parameter_list.append(_compact_args)
            _model_struc_selec_parameter_list.append(_full_args)
        elif _cls == "ER":
            _full_args = get_default_args(getattr(module, _cls))
            _compact_form_keys = [
                "ylag",
                "xlag",
                "estimator",
                "k",
                "n_perm",
                "skip_forward",
                "model_type",
                "basis_function",
            ]
            _compact_args = dict(((key, _full_args[key]) for key in _compact_form_keys))
            _model_struc_selec_parameter_list.append(_compact_args)
            _model_struc_selec_parameter_list.append(_full_args)
        else:
            _model_struc_selec_parameter_list.append(
                get_default_args(getattr(module, _cls))
            )

    return _model_struc_selec_parameter_list


def dict_key_to_list(g_dict):
    return list(g_dict.keys())


def dict_values_to_list(g_dict):
    return list(g_dict.values())


def str_to_class(classname, filepath):
    return getattr(filepath, classname)  # pelo jeito, tem q deixar so filepath


def splitter(string_list):
    splitted_list = list()
    for i in range(len(string_list)):
        splitted_list.append(list(filter(None, re.split(r"\D", string_list[i]))))
        splitted_list[i] = int(splitted_list[i][0])
    return splitted_list


def sorter(string_list):
    splitted_list = splitter(string_list)
    sorted_number_list = sorted(splitted_list)
    sorted_list = [0] * len(string_list)
    for i in range(len(string_list)):
        sorted_list[i] = string_list[splitted_list.index(sorted_number_list[i])]
    return sorted_list


def occurrence_check(prefix, string_list):
    return sum(prefix in s for s in string_list)


def session_state_cut(
    ss_dict, prefix, number_widgets
):  # recebe o dicionario de session state, o prefixo da key e o numero de widgets
    occur_ses_state = list(
        s for s in list(ss_dict) if s.startswith(prefix)
    )  # pegado as ocorrencias da key no dicionario
    sorted_list = sorter(occur_ses_state)  # deixa a lista acima em ordem alfanumérica
    reversed_list = sorted_list[::-1]  # invertendo a lista acima
    dif = (
        occurrence_check(prefix, list(ss_dict)) - number_widgets
    )  # o numero de keys excedentes é a diferença entre o numero de ocorrencias no
    # dict de session state e o numero de widgets
    elements_to_cut = list()  # lista pra retornar as keys excedentes

    for i in range(dif):
        elements_to_cut.append(reversed_list[i])
    return elements_to_cut


def plot_results(
    y=None,
    *,
    yhat=None,
    figsize=(8, 5),
    n=100,
    style="seaborn-v0_8-white",
    facecolor="white",
    title="Free run simulation",
):
    plt.style.use(style)
    plt.rcParams["axes.facecolor"] = facecolor
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.plot(y[:n], c="#1f77b4", alpha=1, marker="o", label="Data", linewidth=1.5)
    ax.plot(yhat[:n], c="#ff7f0e", marker="*", label="Model", linewidth=1.5)

    ax.set_title(title, fontsize=18)
    ax.legend()
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Samples", fontsize=14)
    ax.set_ylabel("y, $\hat{y}$", fontsize=14)

    fig.savefig("temp_figs//results_fig.png", bbox_inches="tight")
    image = Image.open("temp_figs//results_fig.png")
    return image


def plot_residues_correlation(
    data=None,
    *,
    figsize=(8, 5),
    n=100,
    style="seaborn-v0_8-white",
    facecolor="white",
    title="Residual Analysis",
    ylabel="Correlation",
    second_fig=False,
):
    plt.style.use(style)
    plt.rcParams["axes.facecolor"] = facecolor
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.plot(data[0], color="#1f77b4")
    ax.axhspan(data[1], data[2], color="#ccd9ff", alpha=0.5, lw=0)
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_ylim([-1, 1])
    ax.set_title(title, fontsize=18)

    if second_fig == True:
        fig.savefig("temp_figs//residues_fig_2.png", bbox_inches="tight")
        image = Image.open("temp_figs//residues_fig_2.png")
    else:
        fig.savefig("temp_figs//residues_fig_1.png", bbox_inches="tight")
        image = Image.open("temp_figs//residues_fig_1.png")
    return image


def adjust_string(label_string):
    spaced_string = " ".join(label_string.split("_"))
    return spaced_string.capitalize()


def get_acronym(string):
    if string == "R2 score":
        return "R2S"
    else:
        oupt = string[0]

        for i in range(1, len(string)):
            if string[i - 1] == " ":
                oupt += string[i]
        return oupt.upper()


def get_lags_list(n):
    lags = []
    for i in range(1, n + 1):
        lags.append(i)
    return lags


def get_model_eq(model):
    r = pd.DataFrame(
        results(
            model.final_model,
            model.theta,
            model.err,
            model.n_terms,
            err_precision=8,
            dtype="sci",
        ),
        columns=["Regressors", "Parameters", "ERR"],
    )

    model_string = "y_k = "
    for ind in r.index:
        model_string += f"  {float(r.iat[ind, 1]):.2f}*{r.iat[ind, 0]}"
        if len(r.index) != ind + 1:
            model_string += "+"
    model_string = model_string.replace("(", "_{")
    model_string = model_string.replace(")", "}")
    return model_string


def get_chinese_font():
    """获取系统中可用的中文字体"""
    fonts = []
    for f in fm.findSystemFonts():
        try:
            font = fm.FontProperties(fname=f)
            if any(name in font.get_name().lower() for name in ['simhei', 'simsun', 'microsoft yahei', 'dengxian']):
                fonts.append(f)
        except:
            continue
    return fonts[0] if fonts else None


def plot_time_series(input_data, output_data):
    """绘制输入输出时间序列图"""
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(input_data.index, input_data.values, label="Input")
    ax.plot(output_data.index, output_data.values, label="Output")
    
    # 设置字体属性
    font_prop = fm.FontProperties(family=['Microsoft YaHei', 'SimHei', 'SimSun'])
    ax.set_title("Time Series", fontproperties=font_prop)
    ax.set_xlabel("Samples", fontproperties=font_prop)
    ax.set_ylabel("Value", fontproperties=font_prop)
    ax.legend(prop=font_prop)
    
    return fig


def plot_correlation_analysis(input_data, output_data, max_lag):
    """相关性分析和绘图"""
    plt.rcParams['axes.unicode_minus'] = False
    font_prop = fm.FontProperties(family=['Microsoft YaHei', 'SimHei', 'SimSun'])
    
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制自相关图
    acf_values = compute_acf(output_data, max_lag)
    ax1.stem(range(len(acf_values)), acf_values)
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_title("自相关函数", fontproperties=font_prop)
    ax1.set_xlabel("滞后", fontproperties=font_prop)
    ax1.set_ylabel("相关系数", fontproperties=font_prop)
    
    # 绘制互相关图
    ccf_values = compute_ccf(input_data, output_data, max_lag)
    ax2.stem(range(len(ccf_values)), ccf_values)
    ax2.axhline(y=0, color='r', linestyle='-')
    ax2.set_title("互相关函数", fontproperties=font_prop)
    ax2.set_xlabel("滞后", fontproperties=font_prop)
    ax2.set_ylabel("相关系数", fontproperties=font_prop)
    
    plt.tight_layout()
    return fig


def convert_arx_to_continuous(arx_params, Ts=0.1):
    """将ARX模型参数转换为连续时间传递函数参数
    
    Args:
        arx_params: 包含ARX模型参数的字典，格式为：
            {
                'a': [a1, a2, ...],  # 输出系数
                'b': [b0, b1, ...],  # 输入系数
                'delay': d           # 延迟（采样点数）
            }
        Ts: 采样时间，默认0.1秒
        
    Returns:
        dict: 包含连续时间模型参数的字典
            对于FOPDT: {'K': gain, 'tau': time_constant, 'theta': delay}
            对于SOPDT: {'K': gain, 'tau1': time_constant1, 'tau2': time_constant2, 'theta': delay}
    """
    import numpy as np
    from scipy import signal
    
    # 构建离散传递函数的分子分母
    b = np.array(arx_params['b'])
    a = np.array([1.0] + arx_params['a'])
    
    # 创建离散传递函数
    dt_sys = signal.TransferFunction(b, a, dt=Ts)
    
    # 转换为连续时间传递函数
    ct_sys = signal.cont2discrete((dt_sys.num, dt_sys.den), Ts, method='tustin')[0]
    
    # 获取极点和零点
    poles = np.roots(ct_sys[1])
    
    # 计算增益
    K = np.sum(b) / (1 + np.sum(arx_params['a']))
    
    # 计算时滞
    theta = arx_params['delay'] * Ts
    
    if len(poles) == 1:  # FOPDT
        tau = -1.0 / poles[0]
        return {
            'model_type': 'FOPDT',
            'K': K,
            'tau': tau,
            'theta': theta
        }
    else:  # SOPDT
        tau1 = -1.0 / poles[0]
        tau2 = -1.0 / poles[1]
        return {
            'model_type': 'SOPDT',
            'K': K,
            'tau1': tau1,
            'tau2': tau2,
            'theta': theta
        }

def estimate_arx_model(x_data, y_data, na=2, nb=2, Ts=0.1):
    """使用最小二乘法估计ARX模型参数
    
    Args:
        x_data: 输入数据
        y_data: 输出数据
        na: AR阶数
        nb: 输入阶数
        Ts: 采样时间
        
    Returns:
        dict: ARX模型参数
    """
    import numpy as np
    from scipy import signal
    
    # 估计时滞
    corr = signal.correlate(y_data - np.mean(y_data),
                           x_data - np.mean(x_data),
                           mode='full')
    delay = len(corr)//2 - np.argmax(corr)
    delay = max(0, delay)  # 确保延迟非负
    
    # 构建回归矩阵
    N = len(y_data)
    max_lag = max(na, nb)
    phi = np.zeros((N-max_lag, na+nb))
    
    # 填充AR项
    for i in range(na):
        phi[:, i] = -y_data[max_lag-i-1:N-i-1]
    
    # 填充输入项
    for i in range(nb):
        if i + delay < len(x_data):
            phi[:, na+i] = x_data[max_lag-i-1-delay:N-i-1-delay]
    
    # 输出向量
    Y = y_data[max_lag:]
    
    # 最小二乘估计
    theta = np.linalg.lstsq(phi, Y, rcond=None)[0]
    
    # 分离参数
    a_params = theta[:na].tolist()
    b_params = theta[na:].tolist()
    
    return {
        'a': a_params,
        'b': b_params,
        'delay': delay
    }
