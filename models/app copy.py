import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = None


def predict_results(model, new_data):
    with open('./models/Scaler.pkl', 'rb') as f:
        columnsN = new_data.columns
        scaler = pkl.load(f)
        new_data = scaler.transform(new_data) 
        new_data =  pd.DataFrame(new_data,columns=columnsN) # Apply scaling on the test data
        # st.dataframe(new_data, hide_index= True)
        y_pred=model.predict(new_data)
        return y_pred

def take_input(model=None):
    if model is not None:
        with st.form('Select Parameters to test:', clear_on_submit=True):
            features = pkl.load(open(f'./models/Features.pkl', 'rb'))
            # st.write(features)
            keys = [x for x in features.keys()]
            default_value = None
            new_data = dict.fromkeys(keys, default_value)
            c1, c2 = st.columns(2, gap="medium", vertical_alignment="center")
            i = 1
            for feature in features.keys():
                
                if i%2 == 0:
                    with c1:
                        options = features[feature].keys()
                        # st.write(options)
                        if feature in ('ct_value_screening'):
                            tempV1 = st.slider(f"Enter value for {feature.upper()}", min_value=1, max_value=max(list(options)))
                        else:
                            tempV1 = features[feature][st.selectbox(f"Select value for {feature.upper()}", 
                                options = options) ]
                else:
                    with c2:
                        options = features[feature].keys()
                        # st.write(options)
                        if feature in ('ct_value_screening'):
                            tempV1 = st.slider(f"Enter value for {feature.upper()}", min_value=1, max_value=max(list(options)))
                        else:
                            tempV1 = features[feature][st.selectbox(f"Select value for {feature.upper()}", 
                                options = options) ]

                i = i+1

                empL = []
                empL.append(tempV1)
                new_data[feature] = empL
            
            st.session_state['subB'] = st.form_submit_button('Submit')
 
    if 'subB' in st.session_state and st.session_state['subB']:
                # st.write(new_data)
                # st.write(subB) 
        # st.write(pd.DataFrame(new_data), hide_index=True)
        new_data = pd.DataFrame(new_data)
        # columnsN = new_data.columns

        y_pred = predict_results(model, new_data)
        # st.write(y_pred)
        if y_pred[0] == 1:
            st.subheader(f'You have Covid-19')
        else:
            st.subheader(f'You don\'t have Covid-19')
        # st.write(pd.DataFrame(y_pred), hide_index = True)

        # st.subheader(f'You have Covid-19 with likelihood of {prediction*100:.2f}%')
        # else:
        #     st.subheader(f'You don\'t have Covid-19')# likelihood of {100 - prediction*100:.2f}%')    

# @st.cache_resource
def load_model(model, kernel):
    try:
        match model:
            case 'Naive Bayesian':
                st.subheader('Predicted Class:NB')
                st.session_state['model_loaded'] = pkl.load(open(f'./models/NB.pkl', 'rb'))
            case 'Decesion Tree':
                st.session_state['model_loaded'] = pkl.load(open(f'./models/DT.pkl', 'rb'))
            case 'Random Forest':
                st.session_state['model_loaded'] = pkl.load(open(f'./models/RF.pkl', 'rb'))
            case 'SVM':
                match kernel:
                    case 'Linear':
                        st.session_state['model_loaded']= pkl.load(open(f'./models/{model}_linear.pkl', 'rb'))
                    case 'RBF':
                        st.session_state['model_loaded'] = pkl.load(open(f'./models/{model}_rbf.pkl', 'rb'))
                    case 'Polynomial':
                        st.session_state['model_loaded'] = pkl.load(open(f'./models/{model}_poly.pkl', 'rb'))
                    case 'Sigmoidal':
                        st.session_state['model_loaded'] = pkl.load(open(f'./models/{model}_sigmoid.pkl', 'rb'))
        # st.success(st.session_state['model_loaded'])
        # st.success('Model loaded successfully!')
        
        
    except FileNotFoundError:
        st.error('Model not found. Please make sure the model file exists.')
        return None

if st.session_state['model_loaded'] is not None:
    st.subheader(f'Model loaded successfully!: {st.session_state['model_loaded']}')
    take_input(st.session_state['model_loaded'])  
def test_page():
    with st.sidebar:
        st.title('Select Model to test')
        model = st.selectbox('Select Models to test: ', options=['Naive Bayesian', 'Decesion Tree',
                                    'Random Forest', 'SVM'])
        if model == 'SVM':
            kernel = st.selectbox('Select Kernel for SVM: ', options=['Linear', 'RBF', 'Polynomial', 'Sigmoidal'])
        else:
            kernel = 'None'

        b1 = st.button('Load Model', on_click=load_model, args= (model, kernel))

# take_input(st.session_state['model_loaded']) 
test_page()




    