import streamlit as st
from sysidentpy.utils.save_load import load_model
from sysidentpy.utils.display_results import results
from sysidentpy.simulation import SimulateNARMAX
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
import os
import sys
sys.path.insert(1, os.path.dirname(__file__).replace('\pages',''))
import assist.utils as utils
import pandas as pd
import pickle as pk
import numpy as np

root = os.path.join(os.path.dirname(__file__).replace('\pages','')+'\\assist')
path = os.path.join(root, "pagedesign.py")

with open(path, encoding="utf-8") as code:
    c = code.read()
    exec(c, globals())

with st.sidebar: #não funcionase estiver em um arquivo externo
    ''' [![Repo](https://badgen.net/github/release/wilsonrljr/sysidentpy/?icon=github&labelColor=373736&label&color=f47c1c)](https://github.com/wilsonrljr/sysidentpy) ''' 
    st.markdown("<br>",unsafe_allow_html=True)

def add_regressors(regr_index_key, regr_list_key): #adiciona um grupo de regressores ao session_state específico
    regkeys = utils.sorter([regk for regk in list(st.session_state) if regr_index_key in regk]) #por segurança, deixa as keys de cada regressor do grupo em ordem alfanumérica
    nested_regressors = [st.session_state[regk] for regk in regkeys] #pega os valores correspondentes de cada key no dict session_state e coloca na lista do grupo de regressores
    st.session_state[regr_list_key].append(nested_regressors) #faz o append da lista do grupo de regressores na lista 'geral' de regressores

def add_thetas(thetas_index_key, thetas_list_key): #o código é semelhante ao acima, a diferença é a última linha
    thetaskeys = utils.sorter([thetask for thetask in list(st.session_state) if thetas_index_key in thetask])
    nested_thetas = [st.session_state[thetask] for thetask in thetaskeys]
    st.session_state[thetas_list_key] = nested_thetas #aqui é feita uma atribuição ao invés de append

tabn = ['Load Model', 'Simulate Model']

tab5, tab6 = st.tabs(tabn)

with tab5:
    col, esp0, col0 = st.columns([5,1,5])
    
    with col:
        st.file_uploader("Validation Input Data", key='vx_data', help='Upload your CSV file')
        if st.session_state['vx_data'] != None:
            x_valid = pd.read_csv(st.session_state['vx_data'], sep='\t', header=None).to_numpy()

    with col0:
        st.file_uploader("Validation Output Data", key='vy_data', help='Upload your CSV file')
        if st.session_state['vy_data'] != None:
            y_valid = pd.read_csv(st.session_state['vy_data'], sep='\t', header=None).to_numpy()

    with col:
        st.file_uploader('Load the model file', key='model_file')

    if st.session_state['model_file'] != None:
        loaded_model = pk.load(st.session_state['model_file'])

    st.markdown("---")

    if st.session_state['model_file'] != None and st.session_state['vx_data'] != None and st.session_state['vy_data'] != None:
        loaded_model.fit(X=x_valid, y=y_valid)
        yhat_loaded = loaded_model.predict(X=x_valid, y=y_valid)

        r_loaded = pd.DataFrame(results(
            loaded_model.final_model, loaded_model.theta, loaded_model.err,
            loaded_model.n_terms, err_precision=8, dtype='sci'
            ),
        columns=['Regressors', 'Parameters', 'ERR'])
        st.subheader('Model Loaded from file \n')
        st.write(r_loaded)

        with st.expander('Results Plot'):
            st.image(utils.plot_results(y=y_valid, yhat=yhat_loaded, n=1000))
    #     st.write('ha')

with tab6:
    col2, esp1, col3 = st.columns([5,1,5])

    if 'regr_list' not in st.session_state:
        st.session_state['regr_list'] = list() #lista 'geral' para armazenar todos os grupos de regressores
        
    if 'thetas_list' not in st.session_state:
        st.session_state['thetas_list'] = list() #lista para armazenar os thetas referentes a cada grupo de regressores

    with col2:
        st.file_uploader("Test Input Data", key='tx_data', help='Upload your CSV file')
        if st.session_state['tx_data'] != None:
            x_test = pd.read_csv(st.session_state['tx_data'], sep='\t', header=None).to_numpy()

    with col3:
        st.file_uploader("Test Output Data", key='ty_data', help='Upload your CSV file')
        if st.session_state['ty_data'] != None:
            y_test = pd.read_csv(st.session_state['ty_data'], sep='\t', header=None).to_numpy()

    st.markdown("---")

    st.write('Define your model (check the [tutorial](https://sysidentpy.org/examples/simulating_a_predefined_model/))')

    st.write('Insert the regressors codification (the order of the codes must be the same as the one in the tutorial)')
    
    col4, esp2, esp3 = st.columns([3, 1, 10])
    
    with col4:
        st.number_input('Number of regressors', key='regr_numb', min_value=1, value=2)
        
    col5, esp4, col6, esp5 = st.columns([3,2,5,5])      
        
    with col5:
        with st.form('Reg form'):
            for reg_input_number in range(st.session_state['regr_numb']):
                st.number_input('Regressor '+str(reg_input_number+1), key='reg_'+str(reg_input_number+1), min_value=0, value=0)
            add_reg = st.form_submit_button('Add regressors')

            if add_reg:
                add_regressors('reg_', 'regr_list')

        model_code_l = np.array(st.session_state['regr_list'])
        clear_regr = st.button('Clear regressors list')
        if clear_regr:
            st.session_state['regr_list'].clear()

        with st.form('Theta form'):
            for numb_theta in range(model_code_l.shape[0]):
                st.number_input('Parameter value for regressor group '+str(numb_theta+1), key='theta_'+str(numb_theta+1), value=1.0e0, format='%e')
            add_theta = st.form_submit_button('Set Parameters')

            if add_theta:
                add_thetas('theta_', 'thetas_list')
        theta_l = np.array([st.session_state['thetas_list']]).T
        clear_thetas = st.button('Clear parameters list')
        if clear_thetas:
            st.session_state['thetas_list'].clear()

        
    with col6:
        st.caption('Regressors groups')
        for regr_text in st.session_state['regr_list']:
            st.text(str(regr_text))  

    col7, esp6, esp7 = st.columns([3, 1, 5])    
    if st.session_state['tx_data'] != None and st.session_state['ty_data'] !=None:

        with col7:
            if 'steps_ahead' not in st.session_state:
                st.session_state['steps_ahead'] = None
            st.write('Free Run Simulation')
            if st.checkbox('', value=True, key='free_run') is False:
                st.number_input('Steps Ahead', key = 'steps_ahead', min_value=1)
            with st.form('Simulate'):
                st.write('Calculate ERR')
                st.checkbox(' ', key='calc_err', value=True)
                st.write('Estimate parameter')
                st.checkbox(' ', key='estimate_par', value=False)
                st.write('Extended least squares')
                st.checkbox(' ', key='extend_least_squares', value=True)
                st.markdown("---")
        
                st.write()
                sim_model = st.form_submit_button('Simulate the model', help='This button will work only if your data is loaded and the regressors/parameters are properly set!')
        
        
        st.markdown("---")

        if sim_model==True:
            sim = SimulateNARMAX(basis_function=Polynomial(), calculate_err=st.session_state['calc_err'], estimate_parameter=st.session_state['estimate_par'], extended_least_squares=st.session_state['extend_least_squares'])
            yhat_sim = sim.simulate(
                X_test = x_test,
                y_test = y_test,
                model_code = model_code_l,
                theta = theta_l,
                steps_ahead = st.session_state['steps_ahead']
            )

            with st.expander('Results Plot'):
                st.image(utils.plot_results(y=y_test, yhat=yhat_sim, n=1000))

            ee = compute_residues_autocorrelation(y_test, yhat_loaded)
            if x_test.shape[1]==1:
                x1e = compute_cross_correlation(y_test, yhat_loaded, x_test)
            else:
                x1e = compute_cross_correlation(y_test, yhat_loaded, x_test[:, 0])

            with st.expander('Residues Plot'):
                st.image(utils.plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$"))
                st.image(utils.plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$", second_fig=True))