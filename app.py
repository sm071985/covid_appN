import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import pycountry

# def predict_results(model, new_data):
#     with open('./models/Scaler.pkl', 'rb') as f:
#         columnsN = new_data.columns
#         scaler = pkl.load(f)
#         new_data_std = scaler.transform(new_data) 
#         # st.dataframe(new_data_std, hide_index= True)
#         new_data =  pd.DataFrame(new_data_std,columns=columnsN) 
#         st.write("New data scaled")# Apply scaling on the test data
#         y_pred=model.predict(new_data)
#         return y_pred

@st.cache_resource
def model_loader(model, new_data):
    scl = pkl.load(open('./models/Scaler.pkl', 'rb'))
    scaler = scl["stdscaler"]
    max_ct = scl["max_ct"]
    columnsN = new_data.columns
    new_data_std = scaler.transform(new_data) 
        # st.dataframe(new_data_std, hide_index= True)
    new_data =  pd.DataFrame(new_data_std,columns=columnsN) 
    # st.write("New data scaled")# Apply scaling on the test data

    # st.write(f'Using Model: {model}')
    try:
        match model:
            case 'Naive Bayesian':
                load_model = pkl.load(open(f'./models/NB{max_ct}.pkl', 'rb'))
            case 'Decesion Tree':
                load_model = pkl.load(open(f'./models/DT{max_ct}.pkl', 'rb'))
            case 'Random Forest':
                load_model = pkl.load(open(f'./models/RF{max_ct}.pkl', 'rb'))
            case 'SVM (Linear)':
                load_model= pkl.load(open(f'./models/SVM_linear{max_ct}.pkl', 'rb'))
            case 'SVM (RBF)':
                load_model= pkl.load(open(f'./models/SVM_rbf{max_ct}.pkl', 'rb'))
            case 'SVM (Polynomial)':
                load_model= pkl.load(open(f'./models/SVM_poly{max_ct}.pkl', 'rb'))
            case 'SVM (Sigmoidal)':
                load_model= pkl.load(open(f'./models/SVM_sigmoid{max_ct}.pkl', 'rb'))
        
        result = load_model.predict(new_data)
        # st.write(result)
        # result = predict_results(st.session_state.load_model, new_data)        
        if result[0] == 1:
            st.subheader(f'You have Covid-19')
        else:
            st.subheader(f'You don\'t have Covid-19')
        
    except FileNotFoundError:
                st.error('Model not found. Please make sure the model file exists.')


def test_page(model=None):
    load_model = None
    features = pkl.load(open(f'./models/Features.pkl', 'rb'))
    keys = [features[x] for x in features.keys()]
    new_data = pd.DataFrame(columns=keys)

    with st.form('prsnlInfo', clear_on_submit=False):
        st.header("Data Collection")
        c_00, c_01 = st.columns(2, gap="medium", vertical_alignment="top")
        with c_00:
            st.subheader('Personal Information')
            name = st.text_input('Name: ', placeholder='Enter Your Name')#, on_change = check_blank, args=(value,) )
            age = st.slider('Age: (Select 100 if age is more than 100)', min_value=0, max_value=100, step=1, )
            new_data.loc[0,'age'] = age
            sex = st.selectbox('Sex: ', options= ["Male", "Female"],)
            state = st.text_input('State: ', placeholder='Enter the State you are from')#, on_change=check_blank, args = (value,))
            country_list = list(pycountry.countries)
            cn_list = [c.name for c in country_list]
            country = st.selectbox('Country: ', options=cn_list, placeholder='Select your Country')
            expo_infec = st.selectbox('Exposed to infected zone: ', options=["No", "Yes"])
            eff_mem = st.slider('No. of effected family members: (Select 10 if number is more than 10) ', max_value=10, min_value=0)
        # next_sec = st.form_submit_button('Next Section')
        with c_01:
            st.subheader('Clinical Information')
            c_0, c_1 = st.columns(2, gap="medium", vertical_alignment="top")
            i = 0
            for feature, key in features.items():
                if feature not in ('Age','Comorbidity','RTPCR Test(CT VALUE)'): # 'Comorbidity', 
                    match i%2:
                        case 0:
                            with c_0:
                                new_data.loc[0,key] = 1 if st.checkbox(feature, key=key) else 0
                        case 1:
                            with c_1:
                                new_data.loc[0,key] = 1 if st.checkbox(feature, key=key) else 0
                    i += 1

                else:
                    if feature == 'Comorbidity':
                        new_data.loc[0,key] = 1 if st.selectbox(feature, options=["No", "Yes"]) == 'Yes' else 0
                                    
                    if feature == 'RTPCR Test(CT VALUE)':
                        new_data.loc[0,key] = st.slider(feature+'Select 10 if CT value is more than 50', max_value= 50, min_value=0)

        st.header('Model Selection')
        modeli = st.selectbox('Select Model: ', options=['Naive Bayesian', 'Decesion Tree', 'Random Forest',
                            'SVM (Linear)', 'SVM (RBF)', 'SVM (Polynomial)', 'SVM (Sigmoidal)'],)
            # kernel = st.selectbox('Select Kernel: ', options=['Linear', 'RBF', 'Polynomial', 'Sigmoidal'],)
        btn_lm = st.form_submit_button('Predict')#, on_click=model_loader,args=(modeli, pd.DataFrame.from_dict(new_data)))
    if btn_lm:
        st.write("New data raw")
        st.dataframe(new_data)
        model_loader(modeli, new_data)  # Call model_loader function with selected model and new_data
test_page()




    
