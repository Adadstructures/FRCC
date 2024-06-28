import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initialize uploaded_file variable
uploaded_file = None

# Load your models
loaded_cato_rca_strength = pickle.load(open('Trained_CATO_RCA_Strength.sav', 'rb'))
loaded_cato_rca_strain = pickle.load(open('Trained_CATO_RCA_Strain.sav', 'rb'))
loaded_cato_na_strength = pickle.load(open('Trained_CATO_NA_Strength.sav', 'rb'))
loaded_cato_na_strain = pickle.load(open('Trained_CATO_NA_Strain.sav', 'rb'))

# Title of the application
st.title("FRCC Strength and Strain Prediction")

# Aggregate Type Dropdown
aggregate_type = st.selectbox("Aggregate Type", ["RA", "NA"], index=0)

# Sections for user inputs
with st.form("input_form"):
    # Conditional display based on Aggregate Type
    if aggregate_type == 'RA':
        st.subheader("Aggregate Properties")
        percentage_rca = st.number_input("Percentage of RA replacement by weight", value=50.00, format="%.2f")
        max_rca_size = st.number_input("Maximum diameter of the RA (mm)", value=20.00, format="%.2f")

        st.subheader("Mix/Cementitious Properties")
        water_cement_ratio = st.number_input("Water-to-cement ratio", value=0.35, format="%.2f")
        silica_fume = st.number_input("Percentage of silica fume content by weight", value=0.0, format="%.2f")

    st.subheader("Geometry Properties")
    diameter = st.number_input("Diameter of the concrete cylinder (mm)", value=150.00, format="%.2f")
    height = st.number_input("Height of the concrete cylinder (mm)", value=300.00, format="%.2f")

    st.subheader("Concrete Properties")
    unconfined_strength = st.number_input("Unconfined Strength (MPa)", value=40.00, format="%.2f")
    unconfined_strain = st.number_input("Unconfined Strain", value=0.002, format="%.5f")

    st.subheader("FRP Properties")
    fibre_modulus = st.number_input("Fibre Modulus (MPa)", value=246000.0)  # Kept as an integer where decimals aren't necessary
    frp_overall_thickness = st.number_input("FRP Overall Thickness (mm)", value=0.501, format="%.3f")
    rupture_strain = st.number_input("Rupture Strain", value=0.0140, format="%.4f")

    # Calculations in the background
    if aggregate_type == 'RA':
        agg_type = 0
        confinement_stress = 2 * rupture_strain * fibre_modulus * frp_overall_thickness / diameter
        stiffness_ratio = 2 * fibre_modulus * frp_overall_thickness / (diameter * (unconfined_strength / unconfined_strain))
        strain_ratio = rupture_strain / unconfined_strain
        concrete_modulus = 4120 * (unconfined_strength ** 0.5)
    else:
        agg_type = 1
        confinement_stress = 2 * rupture_strain * fibre_modulus * frp_overall_thickness / diameter
        stiffness_ratio = 2 * fibre_modulus * frp_overall_thickness / (diameter * (unconfined_strength / unconfined_strain))
        strain_ratio = rupture_strain / unconfined_strain
        concrete_modulus = 4730 * (unconfined_strength ** 0.5)

    # Button to perform calculation
    predict_button = st.form_submit_button("Predict")

# Perform prediction and store values in session state
if 'predicted_strength' not in st.session_state:
    st.session_state.predicted_strength = None

if predict_button:
    if aggregate_type == 'RA':
        input_data_strength = [agg_type, diameter, percentage_rca, max_rca_size, water_cement_ratio, silica_fume, unconfined_strength, unconfined_strain, fibre_modulus, frp_overall_thickness, rupture_strain, concrete_modulus, confinement_stress, strain_ratio, stiffness_ratio]
        input_data_strain = [agg_type, percentage_rca, max_rca_size, water_cement_ratio, silica_fume, unconfined_strength, unconfined_strain, fibre_modulus, frp_overall_thickness, rupture_strain, concrete_modulus, confinement_stress, strain_ratio, stiffness_ratio]
    else:
        input_data_strength = [diameter, height, unconfined_strength, rupture_strain, fibre_modulus, unconfined_strain, confinement_stress, frp_overall_thickness]
        input_data_strain = [height, unconfined_strength, rupture_strain, fibre_modulus, unconfined_strain, confinement_stress, frp_overall_thickness]

    # Predict using the appropriate model
    strength_prediction = loaded_cato_rca_strength.predict([input_data_strength]) if aggregate_type == 'RA' else loaded_cato_na_strength.predict([input_data_strength])
    strain_prediction = loaded_cato_rca_strain.predict([input_data_strain]) if aggregate_type == 'RA' else loaded_cato_na_strain.predict([input_data_strain])

    # Enhancement ratios
    strength_enhancement_ratio = strength_prediction/unconfined_strength
    strain_enhancement_ratio = strain_prediction/unconfined_strain
    
    # Print prediction results
    st.subheader("Prediction Results")
    st.write(f"Ultimate Strength: {strength_prediction[0]:.3f} MPa")
    st.write(f"Ultimate Strain: {100*strain_prediction[0]:.3f} %")
    st.write(f"Strength Enhancement: {strength_enhancement_ratio[0]:.3f}")
    st.write(f"Strain Enhancement: {strain_enhancement_ratio[0]:.3f}")

    # Store predicted values in session state
    st.session_state.predicted_strength = strength_prediction
    st.session_state.predicted_strain = strain_prediction
    st.session_state.strength_enhancement_ratio = strength_enhancement_ratio
    st.session_state.strain_enhancement_ratio = strain_enhancement_ratio

# Option to upload file for experimental data
uploaded_file = st.file_uploader("Upload Experimental Stress-Strain Data (Optional)", type=["csv"])

# Button to plot stress-strain curve
plot_button = st.button("Plot Stress-Strain Curve")

# Plot stress-strain curves
if plot_button:
    # Plotting the Stress-Strain Curve
    # Modulus values
    f_o = unconfined_strength
    Modulus_1 = concrete_modulus
    Modulus_2 = (st.session_state.predicted_strength - f_o)/st.session_state.predicted_strain

    # Generate data for the plot
    section_1 = np.arange(0, unconfined_strain + (unconfined_strain / 10), unconfined_strain / 10)
    section_2 = np.arange(unconfined_strain, st.session_state.predicted_strain + (unconfined_strain / 10), unconfined_strain / 10)
    strain_values = np.concatenate((section_1, section_2))

    # Stress_values prediction
    k_Correction = st.session_state.predicted_strain/(st.session_state.predicted_strain-unconfined_strain)
    f_not = f_o
    confinement_stiffness = 2 * fibre_modulus * frp_overall_thickness / diameter

    if aggregate_type == 'NA':
        n = 1
    elif aggregate_type == 'RA':
        n = max(0.5, confinement_stiffness / Modulus_2)
    else:
        n = 0.5

    e_n = n*unconfined_strain
    e_not = f_not/Modulus_1
    stress_values = ((((Modulus_1 * e_n) - f_not) * np.exp(-strain_values / e_n)) + f_not + (k_Correction*Modulus_2 * strain_values)) * (1 - np.exp(-strain_values / e_n))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(strain_values * 100, stress_values, label='Hybridized CATO-MZW Model (Predicted)')

    # Plot experimental data if uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        plt.plot(df['Strain'] * 100, df['Stress'], label='Experimental')

    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.title("Stress-Strain Curve")
    plt.legend()
    plt.grid(True)

    # Display plot
    st.pyplot(plt)
    
    st.subheader("Prediction Results")
    st.write(f"Ultimate Strength: {st.session_state.predicted_strength [0]:.3f} MPa")
    st.write(f"Ultimate Strain: {100*st.session_state.predicted_strain [0]:.3f} %")
    st.write(f"Strength Enhancement: {st.session_state.strength_enhancement_ratio[0]:.3f}")
    st.write(f"Strain Enhancement: {st.session_state.strain_enhancement_ratio[0]:.3f}")

