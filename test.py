import streamlit as st
import pandas as pd
import joblib
import base64

# Load your pre-trained model and scaler
scaler = joblib.load('standard_scalar_NN.pkl')
model = joblib.load('KNN_model.pkl')
model_names = ['Neural Network', 'Support Vector Machine', 'Naive Bayes',
               'Decision Tree', 'K-Nearest Neighbors']
models = ['NN_model.pkl', 'svc_model.pkl', 'naive_bayes_model.pkl',
          'decision_tree_model.pkl', 'naive_bayes_model.pkl', 'KNN_model.pkl']

# st.write(joblib.load(model))

st.title('Kidney Disease Predictor', anchor="Anchor")

with st.container(border=True):
    side_bg_ext = 'png'
    side_bg = 'dr.png'
    st.markdown(
        f"""
        <style>
        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
        }}
        </style>
        """,
        unsafe_allow_html=True,
        )

   
    st.subheader('Medical information', divider=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('**Age**', step=1,format="%d",min_value=20,max_value=110)
        blood_pressure = st.number_input('**Blood Pressure**', step=1,format='%d',min_value=60,max_value=250)
        specific_gravity = st.slider(
            "**Specific Gravity**", max_value=2.0, min_value=1.0, step=0.000001)
        albumin = st.slider(
            '**Albumin**', min_value=0.0, max_value=5.5, step=0.5)
        sugar = st.number_input(
            '**Sugar Level (in mmol/L)**', max_value=7.0, min_value=0.0, step=1.0)
        red_blood_cells = st.selectbox(
            '**Red Blood Cell**', ['Normal', 'Abnormal'])
        pus_cell = st.selectbox('**Pus Cell**', ['Normal', 'Abnormal'])
        pus_cell_clumps = st.selectbox(
            '**Packed Cell clumps**', ["Present", "Not Present"])
        bacteria = st.selectbox('**Bacteria**', ["Present", "Not Present"])
        blood_glucose_random = st.number_input(
            '**Blood Glucose Random (in mg/dL)**', max_value=200.0, min_value=0.0, step=1.0)
        blood_urea = st.number_input(
            '**Blood Urea (in mg/dL)**', max_value=100.0, step=1.0)
        serum_creatinine = st.number_input(
            "**Serum Creatinine**", max_value=10.0, step=1.0)
        sodium = st.slider('Sodium', max_value=170.0, min_value=90.0, step=1.0)

    with col2:
        potassium = st.number_input(
            '**Potassium (millimoles per liter)**', max_value=7.0, min_value=3.0, step=1.0)
        haemoglobin = st.number_input(
            '**Haemoglobin**', max_value=20.0, min_value=5.0, step=1.0)
        packed_cell_volume = st.number_input(
            '**Packed Cell Volume**', min_value=30.0, max_value=55.0, step=1.0)
        white_blood_cell_count = st.slider(
            '**White blood cell count**', min_value=3000.0, max_value=15000.0, step=1.0)
        red_blood_cell_count = st.slider(
            '**Red blood cell count**', min_value=2.5, max_value=6.5, step=0.5)
        hypertension = st.selectbox('**Hypertension**', ['Yes', 'No'])
        diabetes_mellitus = st.selectbox(
            '**Diabetes Mellitus**', ["Yes", "No"])
        coronary_artery_disease = st.selectbox(
            '**Coronary Artery Disease**', ["Yes", "No"])
        appetite = st.selectbox('**Appetite**', ['Good', 'Poor'])
        peda_edema = st.selectbox('**Peda Edema**', ["Yes", "No"])
        aanemia = st.selectbox('**Aanemia**', ["Yes", "No"])

    UserDetails = pd.DataFrame({'age': age, 'bp': blood_pressure, 'sg': specific_gravity,
                                'al': albumin, 'su': sugar, 'rbc': red_blood_cells,
                                'pc': pus_cell, 'pcc': pus_cell_clumps, 'ba': bacteria,
                                'bgr': blood_glucose_random, 'bu': blood_urea,
                                'sc': serum_creatinine, 'sod': sodium, 'pot': potassium,
                                'hemo': haemoglobin, 'pcv': packed_cell_volume,
                                'wc': white_blood_cell_count, 'rc': red_blood_cell_count,
                                'htn': hypertension, 'dm': diabetes_mellitus,
                                'cad': coronary_artery_disease, 'appet': appetite,
                                'pe': peda_edema, 'ane': aanemia}, index=[0])

    # Convert categorical columns to numerical using one-hot encoding
    UserDetails_encoded = pd.get_dummies(UserDetails, columns=['rbc', 'pc', 'pcc',
                                                               'ba', 'htn', 'dm',
                                                               'cad', 'appet', 'pe', 'ane'])
    print('-----------------------------describe()---------------------------------')
    print(UserDetails_encoded.describe())
    print('------------------------------corr()---------------------------------')
    print(UserDetails_encoded.corr())
    print('---------------------------------------------------------------')
    import matplotlib.pyplot as plt

    # UserDetails_encoded.hist(bins=20, figsize=(15, 10))
    # plt.show()

    # Ensure all columns are present after one-hot encoding
    for column in UserDetails.columns:
        if column not in UserDetails_encoded.columns:
            UserDetails_encoded[column] = 0

    # Drop the original categorical columns
    UserDetails_encoded = UserDetails_encoded.drop(['rbc', 'pc', 'pcc', 'ba',
                                                    'htn', 'dm', 'cad',
                                                    'appet', 'pe', 'ane'], axis=1)

    button = st.button('submit')
# if button:
#     model_predictions = {}
#     for model_file, model_name in zip(models, model_names):
#         prediction_model = joblib.load(model_file)
#         user_data_scaled = scaler.transform(UserDetails_encoded)
#         prediction = prediction_model.predict(user_data_scaled)
#         model_predictions[model_name] = prediction[0]

#     st.write("**Kidney disease predictions**")
#     for model_name, prediction in model_predictions.items():
#         print(f"{model_name} prediction: {prediction}")
#     count_1 = 0
#     count_0 = 0

#     for value in model_predictions.values():
#         if value == 1:
#             count_1 += 1
#         elif value == 0:
#             count_0 += 1


#     # Determine overall prediction based on majority vote
#     if count_1 > count_0:
#         st.write("Overall prediction: The person  have kidney disease.")
#     else:
        
#         st.write("Overall prediction: The person do to not have kidney disease.")



if button:
        # print("Model parameters:", model.get_params())
        # st.write("Model parameters:", model.get_params())
        Newdataset = pd.read_csv('newdata.csv')
        scaler = joblib.load('standard_scalar_NN.pkl')
        user_data_scaled = scaler.transform(Newdataset)
        ynew=model.predict(user_data_scaled)
        # st.write(ynew)
        print("------",ynew)
        scaler = joblib.load('standard_scalar_NN.pkl')
        # print(UserDetails_encoded)
        # print('-------------------------------------------------')
        # print(UserDetails_encoded.dtypes)
        # print('--------------------------------------------------')
        user_data_scaled = scaler.transform(UserDetails_encoded)
        prediction = model.predict(user_data_scaled)
        # model_prediction.append(prediction[0])
        model_prediction =[]
        # for model in models:
        #     predictionmodel = joblib.load(model)
        #     # Scale the numerical features
        #     user_data_scaled = scaler.transform(UserDetails_encoded)
        #     # Make prediction using the model
        #     prediction = predictionmodel.predict(user_data_scaled)
        #     model_prediction.append(prediction[0])
        #     # Display the prediction result
        # print(model_prediction)
        # # dictionary = {model:model_prediction[0]}
        # print(dictionary)
        st.write("Kidney disease prediction:", prediction)


