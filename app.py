import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Water Leak Detection",
    page_icon="💧",
    layout="centered"
)

# ==================================
# PREDICTION HISTORY
# ==================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []



# ==========================================
# LOAD MODEL
# ==========================================

model_data = joblib.load(
    "random_forest_leak_model (1).pkl"
)

model = model_data["model"]
features = model_data["features"]

# Load scaler if it exists
scaler = model_data.get("scaler", None)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.image(
    "leak-detection.jpg",
    use_container_width=True
)

st.sidebar.title("💧 Water Leak Detection")

menu = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "ℹ️ About",
        "👨‍💻 Developers"
    ]
)


# ==========================================
# ABOUT
# ==========================================

if menu == "ℹ️ About":

    st.title("ℹ️ About the App")

    st.write("""
    ### 💧 Water Leak Detection

    This application uses Machine Learning to predict
    whether a water pipe is leaking.

    The application uses a Random Forest model trained
    using water network data.
    """)

    st.subheader("🔍 How It Works")

    st.write("""
    1. Enter the pipe information.
    2. The data is processed.
    3. The Random Forest model analyzes the data.
    4. The application predicts whether there is a leak.
    """)

    st.info(
        "This application is a Machine Learning prediction "
        "tool and should support, not replace, professional "
        "water network inspection."
    )


# ==========================================
# HOME
# ==========================================

elif menu == "🏠 Home":

    st.title("💧 Water Leak Detection")

    st.write(
        "Enter the pipe information below to check "
        "whether a leak is detected."
    )

    st.divider()


    # ======================================
    # INPUTS
    # ======================================

    st.subheader("🔧 Pipe Information")


    diameter = st.number_input(
        "Pipe Diameter (inch)",
        min_value=0.0,
        value=3.0,
        step=0.1
    )


    roughness = st.number_input(
        "Roughness (mm)",
        min_value=0.0,
        value=0.16,
        step=0.01
    )


    pressure = st.number_input(
        "Pressure (PSI)",
        min_value=0.0,
        value=49.0,
        step=1.0
    )


    flow = st.number_input(
        "Flow (LPM)",
        min_value=0.0,
        value=300.0,
        step=10.0
    )


    material = st.selectbox(
      "Pipe Material",
       ["HDPE", "PVC", "DCIP"]
    )


    material_hdpe = 1 if material == "HDPE" else 0
    material_pvc = 1 if material == "PVC" else 0


    st.write("")


    # ======================================
    # PREDICTION
    # ======================================

    if st.button(
        "🔍 Check for Leak",
        use_container_width=True
    ):

        # Create input DataFrame
        input_data = pd.DataFrame(
            [[
                diameter,
                roughness,
                pressure,
                flow,
                material_hdpe,
                material_pvc
            ]],
            columns=features
        )


        # ==================================
        # SCALE INPUT IF SCALER EXISTS
        # ==================================

        if scaler is not None:

            input_for_prediction = scaler.transform(
                input_data
            )

        else:

            input_for_prediction = input_data


        # ==================================
        # MAKE PREDICTION
        # ==================================

        prediction = model.predict(
            input_for_prediction
        )[0]


        # ==================================
        # PREDICTION PROBABILITY
        # ==================================
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                input_for_prediction
            )[0]

            # Probability of Leak (class 1)
            probability = probabilities[1]

            # ==========================================
            # RISK LEVEL
            # ==========================================

            if probability < 0.30:
                risk_level = "LOW"
                risk_icon = "🟢"

            elif probability < 0.70:
                risk_level = "MEDIUM"
                risk_icon = "🟠"

            else:
                risk_level = "HIGH"
                risk_icon = "🔴"

            # ==================================
            # SAVE PREDICTION HISTORY
            # ==================================

            history_record = {
               "Prediction": "Leak" if prediction == 1 else "No Leak",
               "No Leak": f"{probabilities[0] * 100:.2f}%",
               "Leak": f"{probabilities[1] * 100:.2f}%",
               "Risk": risk_level,
               "Diameter (Inch)": diameter,
               "Roughness (mm)": roughness,
               "Pressure (PSI)": pressure,
               "Flow (LPM)": flow,
               "Material": material
            }
 
            st.session_state.prediction_history.append(
              history_record
            )


            # ==================================
            # RESULT
            # ==================================

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

              st.error("🚨 LEAK DETECTED")

              # ==================================
              # RECOMMENDED ACTION
              # ==================================

              with st.expander("🛠️ Recommended Action", expanded=True):

                  st.warning("""
                  **Recommended Action**

                  Inspect the pipe and its joints for possible leakage.

                  • Check the current pressure and flow conditions.
                  • Inspect for visible water leakage or pipe damage.
                  • Check nearby pipe joints and connections.
                  • Consider a physical inspection if the abnormal condition continues.
                """)

            else:

              st.success("✅ NO LEAK DETECTED")

              # ==================================
              # RECOMMENDED ACTION
              # ==================================

              with st.expander("🛠️ Recommended Action", expanded=True):

                  st.info("""
                  **Recommended Action**

                  Continue normal monitoring of the pipe.

                  • Record the current pressure and flow readings.
                  • Use the current readings as a baseline.
                  • Monitor for unusual changes in pressure or flow.
                """)


            # ==================================
            # PREDICTION CONFIDENCE
            # ==================================
        
            st.subheader("Prediction Confidence")

            col1, col2 = st.columns(2)

            with col1:

              st.metric(
                 "No Leak",
                 f"{probabilities[0] * 100:.2f}%"
                )

            with col2:

              st.metric(
                 "Leak",
                  f"{probabilities[1] * 100:.2f}%"
               )

        # ==================================
        # RISK LEVEL
        # ==================================

        st.subheader("Risk Level")

        st.metric(
          "Current Risk",
           f"{risk_icon} {risk_level}"
        )


        # ==================================
        # INPUT GRAPH
        # ==================================

        st.subheader("📊 Your Input Values")

        input_graph = pd.DataFrame({
            "Feature": [
                "Diameter (Inch)",
                "Roughness (mm)",
                "Pressure (PSI)",
                "Flow (LPM)"
            ],
            "Value": [
                diameter,
                roughness,
                pressure,
                flow
            ]
        })

        st.bar_chart(
            input_graph.set_index("Feature")
        )


        # ==================================
        # PIPE MATERIAL
        # ==================================

        st.write(f"**Pipe Material:** {material}")


        # ==================================
        # PREDICTION HISTORY
        # ==================================

        st.divider()

        st.subheader("📜 Prediction History")

        if st.session_state.prediction_history:

           history_df = pd.DataFrame(
             st.session_state.prediction_history
            )

           st.dataframe(
              history_df,
              use_container_width=True,
              hide_index=True
            )

        else:

          st.info("No predictions have been made yet.")


        # ==================================
        # CLEAR HISTORY
        # ==================================

        if st.button("🗑️ Clear History"):

          st.session_state.prediction_history = []

          st.rerun()
# ==========================================
# DEVELOPERS
# ==========================================

elif menu == "👨‍💻 Developers":

    st.title("👨‍💻 Developers")

    st.write(
        "### Water Leak Detection Project"
    )

    st.write(
        "Developed by:"
    )

    st.divider()

    st.write("**TE Madondo** — 22540776")

    st.write("**M Mthobisi** — 22547937")

    st.write("**L.S Jobe** — 22552331")

    st.write("**A.S Mdaki** — 22540839")

    st.divider()

    st.caption(
        "Water Leak Detection • Machine Learning Project"
    )

with st.sidebar:

    # ==================================
    # DATASET DOWNLOAD
    # ==================================

    st.subheader("📥 Download Dataset")

    st.write(
        "Download the dataset used by the water leak "
        "detection system."
    )

    # ----------------------------------
    # UN-CLEANED DATASET
    # ----------------------------------

    with open(
        "Water_Pipe_Leak.csv",
        "rb"
    ) as file:

        uncleaned_data = file.read()

    st.download_button(
        label="📥 Download Uncleaned Dataset",
        data=uncleaned_data,
        file_name="Water_Pipe_Leak.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ----------------------------------
    # CLEANED DATASET
    # ----------------------------------

    with open(
        "Water_Pipe_Leak_Cleaned.csv",
        "rb"
    ) as file:

        cleaned_data = file.read()

    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=cleaned_data,
        file_name="Water_Pipe_Leak_Cleaned.csv",
        mime="text/csv",
        use_container_width=True
    )