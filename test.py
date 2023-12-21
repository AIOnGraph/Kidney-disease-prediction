import streamlit as st
import pandas as pd
import joblib

# Load your pre-trained model and scaler
scaler = joblib.load('standard_scalar_NN.pkl')
model = joblib.load('decision_tree_model.pkl')
model_names = ['Neural Network', 'Support Vector Machine', 'Naive Bayes',
               'Decision Tree', 'K-Nearest Neighbors']
models = ['NN_model.pkl', 'svc_model.pkl', 'naive_bayes_model.pkl',
          'decision_tree_model.pkl', 'naive_bayes_model.pkl', 'KNN_model.pkl']

# st.write(joblib.load(model))

st.title('Kidney Disease Predictor', anchor="Anchor")

with st.container(border=True):
    st.subheader('Medical information', divider=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('**Age**', step=1.0)
        blood_pressure = st.number_input('**Blood Pressure**', step=1.0)
        specific_gravity = st.number_input(
            "**Specific Gravity**", max_value=1.030, min_value=1.005, step=0.001)
        albumin = st.number_input(
            '**Albumin**', min_value=2.0, max_value=5.5, step=0.5)
        sugar = st.number_input(
            '**Sugar Level**', max_value=5.0, min_value=1.0, step=1.0)
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
            '**Haemoglobin**', max_value=20.0, min_value=7.0, step=1.0)
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

    UserDetails = pd.DataFrame({'Age': age, 'blood_pressure': blood_pressure, 'specific_gravity': specific_gravity,
                                'albumin': albumin, 'sugar': sugar, 'red_blood_cells': red_blood_cells,
                                'pus_cell': pus_cell, 'pus_cell_clumps': pus_cell_clumps, 'bacteria': bacteria,
                                'blood_glucose_random': blood_glucose_random, 'blood_urea': blood_urea,
                                'serum_creatinine': serum_creatinine, 'sodium': sodium, 'potassium': potassium,
                                'haemoglobin': haemoglobin, 'packed_cell_volume': packed_cell_volume,
                                'white_blood_cell_count': white_blood_cell_count, 'red_blood_cell_count': red_blood_cell_count,
                                'hypertension': hypertension, 'diabetes_mellitus': diabetes_mellitus,
                                'coronary_artery_disease': coronary_artery_disease, 'appetite': appetite,
                                'peda_edema': peda_edema, 'aanemia': aanemia}, index=[0])

    # Convert categorical columns to numerical using one-hot encoding
    UserDetails_encoded = pd.get_dummies(UserDetails, columns=['red_blood_cells', 'pus_cell', 'pus_cell_clumps',
                                                               'bacteria', 'hypertension', 'diabetes_mellitus',
                                                               'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'])

    # Ensure all columns are present after one-hot encoding
    for column in UserDetails.columns:
        if column not in UserDetails_encoded.columns:
            UserDetails_encoded[column] = 0

    # Drop the original categorical columns
    UserDetails_encoded = UserDetails_encoded.drop(['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                                                    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                                                    'appetite', 'peda_edema', 'aanemia'], axis=1)

    button = st.button('submit')
if button:
    model_predictions = {}
    for model_file, model_name in zip(models, model_names):
        prediction_model = joblib.load(model_file)
        user_data_scaled = scaler.transform(UserDetails_encoded)
        prediction = prediction_model.predict(user_data_scaled)
        model_predictions[model_name] = prediction[0]

    st.write("**Kidney disease predictions**")
    # for model_name, prediction in model_predictions.items():
    #     st.write(f"{model_name} prediction: {prediction}")
    count_1 = 0
    count_0 = 0

    for value in model_predictions.values():
        if value == 1:
            count_1 += 1
        elif value == 0:
            count_0 += 1


    # Determine overall prediction based on majority vote
    if count_1 > count_0:
        st.write("Overall prediction: The person  have kidney disease.")
    else:
        
        st.write("Overall prediction: The person do to not have kidney disease.")