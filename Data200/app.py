import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, 
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score,
    f1_score
)

st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
st.title("📊 Model Evaluation & Infographics")
st.write("Upload your testing CSV data to automatically generate feature pipelines, evaluate the model, and view performance graphs.")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your Testing Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Required base columns to correctly recreate the pipeline features
    required_raw_cols = [
        'datetime', 'precipitation', 'humidity', 'cloudcover', 'temp', 'dew', 
        'tempmax', 'tempmin', 'windspeed', 'winddir', 'sealevelpressure', 
        'sunset', 'sunrise', 'visibility', 'solarradiation', 'solarenergy', 'uvindex'
    ]
    
    missing_cols = [col for col in required_raw_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"The uploaded CSV is missing columns required for the notebook's feature engineering steps: {missing_cols}")
    else:
        # 2. Load the Model
        @st.cache_resource
        def load_model():
            return joblib.load("xgb_model.joblib")
            
        try:
            model = load_model()
            st.success("✅ Model `xgb_model.joblib` successfully connected and matched!")
        except Exception as e:
            st.error(f"Error loading model: {e}. Please ensure `xgb_model.joblib` is in the same directory as this script.")
            st.stop()
            
        if st.button("Generate Infographics"):
            try:
                # Parse datetime and sort to ensure rolling/lag time-series logic holds true
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime').reset_index(drop=True)
                
                # --- Notebook Feature Engineering Pipeline Replicated ---
                # Lagging features
                df['precip_lag1'] = df['precipitation'].shift(1)
                df['precip_lag3'] = df['precipitation'].shift(3)
                df['precip_lag7'] = df['precipitation'].shift(7)

                df['humidity_lag1'] = df['humidity'].shift(1)
                df['cloudcover_lag1'] = df['cloudcover'].shift(1)

                # Rolling statistics
                df['precip_mean_7d'] = df['precipitation'].rolling(7).mean()
                df['precip_mean_30d'] = df['precipitation'].rolling(30).mean()

                df['humidity_mean_7d'] = df['humidity'].rolling(7).mean()
                df['temp_mean_7d'] = df['temp'].rolling(7).mean()

                # Cyclic date features
                df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)

                df['day_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear / 365)
                df['day_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)

                # Combination indicators
                df['dew_gap'] = df['temp'] - df['dew']
                df['temp_range'] = df['tempmax'] - df['tempmin']
                df['wind_v'] = df['windspeed'] * np.sin(np.radians(df['winddir']))
                df['cloud_mean_3d'] = df['cloudcover'].rolling(3).mean()
                df['humidity_change'] = df['humidity'].diff()
                df['humid_pressure'] = df['humidity'] / df['sealevelpressure']

                # Day length calculations
                df['day_length'] = (
                    pd.to_datetime(df['sunset']) - pd.to_datetime(df['sunrise'])
                ).dt.total_seconds() / 3600
                
                # Exact 32 features in the exact order requested by the model
                features_order = [
                    'tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'winddir', 'sealevelpressure', 
                    'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 
                    'precip_lag1', 'precip_lag3', 'precip_lag7', 'humidity_lag1', 'cloudcover_lag1', 
                    'precip_mean_7d', 'precip_mean_30d', 'humidity_mean_7d', 'temp_mean_7d', 
                    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'dew_gap', 'temp_range', 
                    'wind_v', 'cloud_mean_3d', 'humidity_change', 'humid_pressure', 'day_length'
                ]
                
                # Fill initial rolling/lagging NaN gaps with backfill/forwardfill to prevent inference drops
                df_features = df.copy()
                df_features[features_order] = df_features[features_order].bfill().ffill()
                
                X = df_features[features_order]
                y_true = df_features['precipitation']
                
                # 3. Predict Continuous Values
                predictions = model.predict(X)
                
                # Save into an evaluation dataframe
                eval_data = pd.DataFrame({
                    'datetime': df_features['datetime'],
                    'Actual': y_true,
                    'Predicted': predictions
                })
                
                # Apply the notebook's classification threshold (Rain >= 1mm)
                threshold = 1.0
                eval_data['Actual_Rain'] = eval_data['Actual'] >= threshold
                eval_data['Predicted_Rain'] = eval_data['Predicted'] >= threshold
                
                # --- Display Global Metrics ---
                st.markdown("---")
                st.subheader("📊 Global Performance Metrics")
                
                mae = mean_absolute_error(eval_data['Actual'], eval_data['Predicted'])
                acc = accuracy_score(eval_data['Actual_Rain'], eval_data['Predicted_Rain'])
                prec = precision_score(eval_data['Actual_Rain'], eval_data['Predicted_Rain'], zero_division=0)
                rec = recall_score(eval_data['Actual_Rain'], eval_data['Predicted_Rain'], zero_division=0)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Regression MAE", f"{mae:.4f}")
                c2.metric("Rain Accuracy", f"{acc:.4f}")
                c3.metric("Rain Precision", f"{prec:.4f}")
                c4.metric("Rain Recall", f"{rec:.4f}")
                
                # Confusion Matrix
                st.write("**Confusion Matrix (Rain Class Classification)**")
                cm = confusion_matrix(eval_data['Actual_Rain'], eval_data['Predicted_Rain'])
                fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax_cm,
                            xticklabels=['No Rain (<1mm)', 'Rain (>=1mm)'], 
                            yticklabels=['No Rain (<1mm)', 'Rain (>=1mm)'])
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                st.pyplot(fig_cm)
                
                # --- Rolling 3-Month Window Aggregations ---
                st.markdown("---")
                st.subheader("📅 3-Month Window Metrics (Best vs. Worst Periods)")
                
                eval_data.set_index('datetime', inplace=True)
                
                # Calculate errors & correctness over time
                eval_data['Error'] = (eval_data['Actual'] - eval_data['Predicted']).abs()
                monthly_mae = eval_data['Error'].resample('ME').mean()
                rolling_mae = monthly_mae.rolling(window=3).mean().dropna()
                
                eval_data['Correct'] = (eval_data['Actual_Rain'] == eval_data['Predicted_Rain']).astype(int)
                monthly_acc = eval_data['Correct'].resample('ME').mean()
                rolling_acc = monthly_acc.rolling(window=3).mean().dropna()
                
                if not rolling_mae.empty:
                    best_mae_date = rolling_mae.idxmin()
                    worst_mae_date = rolling_mae.idxmax()
                    best_acc_date = rolling_acc.idxmax()
                    worst_acc_date = rolling_acc.idxmin()
                    
                    col_b, col_w = st.columns(2)
                    with col_b:
                        st.success(f"🟢 **Best 3-Month MAE Window:** Ending {best_mae_date.strftime('%B %Y')} (MAE: {rolling_mae.loc[best_mae_date]:.4f})")
                        st.success(f"🟢 **Best 3-Month Accuracy Window:** Ending {best_acc_date.strftime('%B %Y')} (Accuracy: {rolling_acc.loc[best_acc_date]:.4f})")
                    with col_w:
                        st.warning(f"🔴 **Worst 3-Month MAE Window:** Ending {worst_mae_date.strftime('%B %Y')} (MAE: {rolling_mae.loc[worst_mae_date]:.4f})")
                        st.warning(f"🔴 **Worst 3-Month Accuracy Window:** Ending {worst_acc_date.strftime('%B %Y')} (Accuracy: {rolling_acc.loc[worst_acc_date]:.4f})")
                    
                    # Plot final infographic charts
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
                    
                    # MAE Infographic Plot
                    ax1.plot(rolling_mae.index, rolling_mae.values, marker='o', color='#1f77b4', label='3-Mo Rolling MAE')
                    ax1.axvline(best_mae_date, color='green', linestyle='--', linewidth=2, label='Best Window (Min MAE)')
                    ax1.axvline(worst_mae_date, color='red', linestyle='--', linewidth=2, label='Worst Window (Max MAE)')
                    ax1.set_title("3-Month Rolling Mean Absolute Error (MAE)", fontsize=12)
                    ax1.set_ylabel("MAE Value")
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    ax1.legend()
                    
                    # Accuracy Infographic Plot
                    ax2.plot(rolling_acc.index, rolling_acc.values, marker='s', color='#ff7f0e', label='3-Mo Rolling Accuracy')
                    ax2.axvline(best_acc_date, color='green', linestyle='--', linewidth=2, label='Best Window (Max Acc)')
                    ax2.axvline(worst_acc_date, color='red', linestyle='--', linewidth=2, label='Worst Window (Min Acc)')
                    ax2.set_title("3-Month Rolling Prediction Accuracy", fontsize=12)
                    ax2.set_xlabel("Timeline")
                    ax2.set_ylabel("Accuracy Rate")
                    ax2.grid(True, linestyle='--', alpha=0.5)
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("⚠️ *The dataset timeline doesn't encompass a wide enough range to yield a 3-month rolling average.*")
                    
            except Exception as pipeline_err:
                st.error(f"Error during execution pipeline calculations: {pipeline_err}")