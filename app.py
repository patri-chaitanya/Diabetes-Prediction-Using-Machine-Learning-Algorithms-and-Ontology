import streamlit as st
import pandas as pd
import numpy as np
from src.visualizations import DataVisualizer
from src.styles import TITLE_STYLE, SIDEBAR_STYLE
from src.streamlit_utils import DataContent, DataTable
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import io
from sklearn.preprocessing import RobustScaler
import joblib


st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",  # Expands content area
    initial_sidebar_state="expanded",  # Keeps sidebar open
)

def convert_df_to_csv(df):
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown('<h1 class="styled-title">Diabetes Prediction Application</h1>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">Select Options</div>', unsafe_allow_html=True)
    

    if 'page' not in st.session_state:
        st.session_state['page'] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None 
    
    if 'pre_df' not in st.session_state:
        st.session_state.pre_df = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state['page'] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state['page'] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state['page'] = "Sample Training Data"

    if st.sidebar.button("Know About Data"):
        st.session_state['page'] = "Know About Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state['page'] = "Data Preprocessing"

    if st.sidebar.button("Exploratory Data Analysis"):
        st.session_state['page'] = "Exploratory Data Analysis"

    if st.sidebar.button("Machine Learning Models Used"):
        st.session_state['page'] = "Machine Learning Models Used"

    if st.sidebar.button("Model Predictions"):
        st.session_state['page'] = "Model Predictions"

################################################################################################################

    if st.session_state['page']== "Problem Statement":
        st.image("./Sentiment-analysis-marketing-Final.jpg", width=500)
        st.markdown(DataContent.problem_statement)
    
    elif  st.session_state['page'] == "Project Data Description":
        st.markdown(DataContent.project_data_details)

    elif st.session_state['page'] == "Sample Training Data":
        st.markdown("## 📊 Training Data Preview")
        st.write("🔍 Below is an **interactive table** displaying the first 100 rows:")
        file_path = (r"C:\Users\PATRI CHAITANYA\AppData\Local\Programs\Python\Python310\Batch -12\data\dataset\diabetes.csv")
        st.session_state.df = pd.read_csv(file_path)
        data_table = DataTable(df=st.session_state.df)
        data_table.display_table()


    elif  st.session_state['page'] == "Know About Data":
        file_path = (r"C:\Users\PATRI CHAITANYA\AppData\Local\Programs\Python\Python310\Batch -12\data\dataset\diabetes.csv")
        st.session_state.df = pd.read_csv(file_path)
        st.header("Data Information")

        if "profile_report_generated" not in st.session_state:
            with st.status("⏳ Generating Overall Data Profile Report...", expanded=True) as status:
                profile = ProfileReport(st.session_state.df, explorative=True)
                profile.to_file("ydata_profiling_report.html")
                st.session_state["profile_report_generated"] = True  # Mark as generated
                status.update(label="✅ Report Generated Successfully!", state="complete")

        try:
            with open("ydata_profiling_report.html", "r", encoding="utf-8") as f:
                report_html = f.read()
            html(report_html, height=1000,width=800, scrolling=True)  

        except FileNotFoundError:
            st.error("Report file not found. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    elif  st.session_state['page'] == "Data Preprocessing":
        st.markdown(DataContent.Data_preprocessing)
        pre_df_file = (r"C:\Users\PATRI CHAITANYA\AppData\Local\Programs\Python\Python310\Batch -12\data\dataset\diabetes.csv")
        st.session_state.pre_df = pd.read_csv(pre_df_file)
        st.write("### Preprocessed Data Preview (First 15 Rows)")
        data_table = DataTable(df=st.session_state.pre_df.head(15))
        data_table.display_table()


    elif st.session_state['page'] == "Exploratory Data Analysis":
        file_path = (r"C:\Users\PATRI CHAITANYA\AppData\Local\Programs\Python\Python310\Batch -12\data\dataset\diabetes.csv")
        st.session_state.df = pd.read_csv(file_path)
        st.header("Data Visualization")
        visualizer = DataVisualizer()
        
        plot_type = st.selectbox("Select Visualization", 
            ["Correlation Heatmap", "Feature Boxplots", "Feature Histograms", "Pairplot"])
        
        if plot_type == "Correlation Heatmap":
            with st.spinner("Generating Correlation Heatmap..."):
                fig = visualizer.plot_correlation_heatmap(st.session_state.df)
                st.plotly_chart(fig, use_container_width=False)
        
        elif plot_type == "Feature Boxplots":
            with st.spinner("Generating Box Plots..."):
                fig = visualizer.plot_boxplots(st.session_state.df)
                st.plotly_chart(fig, use_container_width=False)
        
        elif plot_type == "Feature Histograms":
            with st.spinner("Generating Histograms..."):
                fig = visualizer.plot_histograms(st.session_state.df)
                st.plotly_chart(fig, use_container_width=False)
        
        elif plot_type == "Pairplot":
            with st.spinner("Generating Pairplot..."):
                fig = visualizer.plot_pairplot(st.session_state.df)
                scrollable_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    </head>
                    <body>
                        <div style="overflow-x: auto; width: 100%;">
                            {fig.to_html(full_html=False)}
                        </div>
                    </body>
                    </html>
                    """
                components.html(scrollable_html, height=2600, scrolling=True)

    
    elif st.session_state['page'] == "Machine Learning Models Used":
        st.markdown(DataContent.ml_models)
        df_metrics = pd.read_csv(r"C:\Users\PATRI CHAITANYA\AppData\Local\Programs\Python\Python310\Batch -12\data\dataset\diabetes.csv")
        data_table = DataTable(df=df_metrics)
        data_table.display_table()
        st.markdown(DataContent.best_model)
       
    
    elif st.session_state['page'] == "Model Predictions":

        upload_type = ("Upload File")

        if upload_type == "Upload File":
            uploaded_file = st.file_uploader("Choose a CSV or Excel File", type=['csv', 'xlsx'])

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)

                    st.write("Uploaded Data Preview")
                    st.dataframe(data.head())

                    st.success("File Uploaded Successfully")

                    # Load Model using joblib
                    model_file = 'model.pkl'  # Your model file
                    model = joblib.load(open(model_file, 'rb'))

                    if st.button("Load Model & Predict"):
                        with st.spinner("Preprocessing Data..."):

                            # ✅ Preprocessing Steps
                            data_copy = data.copy()

                            # Replace 0 with NaN in selected columns
                            zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
                            data_copy[zero_cols] = data_copy[zero_cols].replace(0, np.nan)

                            # Load Original Dataset (Where Outcome is Present) for Median Calculation
                            original_data = pd.read_csv(r'C:\Users\PATRI CHAITANYA\AppData\Local\Programs\Python\Python310\Batch -12\data\dataset\diabetes.csv')  # Your original dataset with Outcome column

                            # Median Function
                            def median_target(var):
                                temp = original_data[original_data[var].notnull()]
                                return temp.groupby(['Outcome'])[var].median()

                            # Fill NaN with Median Based on Outcome
                            for col in zero_cols:
                                medians = median_target(col)
                                data_copy.loc[data_copy[col].isnull(), col] = medians.mean()

                            # Feature Engineering
                            data_copy["NewBMI"] = pd.cut(data_copy["BMI"], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
                                                        labels=["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"])

                            data_copy["NewInsulinScore"] = data_copy["Insulin"].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")
                            data_copy["NewGlucose"] = pd.cut(data_copy["Glucose"], bins=[0, 70, 99, 126, float('inf')],
                                                            labels=["Low", "Normal", "Above Normal", "High"])

                            # Encoding
                            data_copy = pd.get_dummies(data_copy, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

                            # Scaling
                            scaler = RobustScaler()
                            numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
                            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])

                            st.success("Data Preprocessed Successfully")

                        with st.spinner("Making Predictions..."):
                            model_input_cols = model.feature_names_in_

                            # Automatically Create Missing Dummies
                            missing_cols = [col for col in model_input_cols if col not in data_copy.columns]
                            for col in missing_cols:
                                data_copy[col] = 0

                            # Fill Remaining NaN
                            data_copy.fillna(0, inplace=True)

                            # Select Only Model Input Columns
                            data_copy = data_copy[model_input_cols]

                            # Predict
                            predictions = model.predict(data_copy)

                            # Append Predictions to Original Data
                            data['Predictions'] = predictions

                            # ✅ Convert Outcome Labels
                            data['Outcome_Label'] = data['Predictions'].map({1: 'Diabetic', 0: 'Non-Diabetic'})

                            st.subheader("Sample Predictions")
                            st.dataframe(data.head(10))

                            # Download Predictions File
                            csv = data.to_csv(index=False)
                            st.download_button("Download Predictions", csv, file_name="predictions.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"Error processing file: {e}")



if __name__ == "__main__":
    main()
    