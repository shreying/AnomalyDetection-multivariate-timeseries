# Time Series Anomaly Detection Pipeline

A powerful, ensemble-based anomaly detection system for multivariate time series data. This pipeline combines a **LSTM Autoencoder** for capturing temporal patterns with a suite of **PyOD statistical models** for robust outlier detection, providing interpretable results with feature attribution.

## 🧠 What This Code Does

This project is designed to identify unusual patterns and potential faults in time-series data (e.g., server metrics, sensor readings, financial data). It goes beyond simple anomaly scoring by:

1.  **Preprocessing:** Cleanses and scales your time-series data, handling missing values and validating data quality.
2.  **Feature Extraction:** Uses an LSTM Autoencoder to learn the normal, underlying patterns and relationships within the data. The reconstruction error from this model becomes a powerful feature for detecting anomalies.
3.  **Ensemble Detection:** Combines the strengths of multiple anomaly detection algorithms:
    *   **LSTM Autoencoder (Telemanom-style):** Excellent at detecting complex temporal deviations.
    *   **Isolation Forest (IForest):** Efficient at isolating point anomalies.
    *   **ECOD & COPOD:** Advanced statistical models that are parameter-free and fast.
4.  **Interpretable Results:** Not only flags anomalies but also identifies the **top contributing features** for each anomaly, helping you understand the "why" behind the alert.
5.  **Dynamic Thresholding & Visualization:** Applies statistical thresholds to classify anomalies into severity levels (Normal, Moderate, Significant, Severe) and generates a clear plot to visualize results over time.

### Architecture Overview



## ⚙️ How It Works

1.  **Data Loading & Cleaning (`DataProcessor`):**
    *   Loads the CSV, parses the `Time` column, and sorts the data.
    *   Validates minimum data length and handles missing values via interpolation.
    *   Removes constant features that provide no signal for detection.
    *   Applies Standard Scaling to normalize all numerical features.

2.  **Temporal Feature Learning (`LSTMAutoencoder`):**
    *   The data is windowed into sequences (e.g., 24 hours).
    *   An autoencoder is trained to reconstruct these sequences. Well-reconstructed sequences are considered "normal".
    *   The **Mean Squared Error (MSE)** between the original and reconstructed sequences is calculated, forming the primary LSTM anomaly score.

3.  **Ensemble Anomaly Scoring (`AnomalyDetector`):**
    *   The LSTM reconstruction errors are normalized to a 0-100 scale.
    *   The PyOD models are trained on the same data and their scores are also normalized.
    *   A **weighted ensemble score** is computed (default: 50% LSTM, 20% IForest, 15% ECOD, 15% COPOD), creating a robust final anomaly score.

4.  **Feature Attribution:**
    *   For each data point, the code analyzes which features contributed most to the reconstruction error.
    *   Only features with >1% contribution are considered.
    *   The top 7 features are listed in the output, providing crucial context for each anomaly.

5.  **Classification & Output:**
    *   Scores are classified into static levels: `Normal` (0-10), `Slightly Unusual` (10-30), `Moderate` (30-60), `Significant` (60-90), `Severe` (90-100).
    *   Results are saved to a CSV containing all original data, scores, anomaly levels, and top features.
    *   A dynamic plot is generated with statistical thresholds, showing the smoothed score and highlighting anomalies.

## 🚀 Sample Usage

### Prerequisites

```bash
pip install pyod tensorflow scikit-learn matplotlib seaborn pandas numpy
```

### Basic Python Script

```python
from anomaly_detection_pipeline import run_pipeline

# Run the complete pipeline
df_results = run_pipeline(
    input_csv="server_metrics.csv",
    output_csv="anomaly_results.csv",
    train_end="2023-10-01 23:59:00",  # End of training period (normal data)
    full_end="2023-10-15 07:59:00"    # End of the period to analyze
)

print("Analysis Complete! Results saved to anomaly_results.csv")
print(df_results[['Time', 'Abnormality_score', 'Anomaly_Level']].head())
```

### Google Colab / Jupyter Notebook

```python
# Install dependencies first
!pip install pyod tensorflow scikit-learn matplotlib seaborn

from google.colab import drive
drive.mount('/content/drive')

from anomaly_detection_pipeline import run_pipeline

input_path = "/content/drive/MyDrive/data/server_metrics.csv"
output_path = "/content/drive/MyDrive/data/anomaly_results.csv"

df_final = run_pipeline(input_path, output_path)
df_final.head()
```

### Command Line Interface (CLI)

```bash
# Run with default parameters
python anomaly_detection_pipeline.py --input data/metrics.csv --output results/anomalies.csv

# Run with custom time ranges and training data
python anomaly_detection_pipeline.py \
    --input data/metrics.csv \
    --output results/anomalies.csv \
    --train_end "2023-10-05 23:59:00" \
    --full_end "2023-10-19 07:59:00"
```

## 📁 Input & Output

### Expected Input Format
Your CSV file must contain a `Time` column and various numerical features.
```csv
Time,CPU_Usage,Memory_MB,Network_KBps,Disk_IOps
2023-10-01 00:00:00,12.5,800.2,102.3,5.1
2023-10-01 01:00:00,13.2,810.5,101.8,5.4
2023-10-01 02:00:00,12.8,805.1,103.1,12.8  # <-- An anomaly in Disk_IOps
```

### Output Results
The output CSV includes all original data plus the following columns:
| Column | Description |
| :--- | :--- |
| `Abnormality_score` | Final ensemble score (0-100). Higher = more anomalous. |
| `Anomaly_Level` | Classification: `Normal`, `Slightly Unusual`, `Moderate`, `Significant`, `Severe`. |
| `top_feature_1` to `top_feature_7` | The features that contributed most to the anomaly, ordered by importance. |

## 📊 Example Output Visualization

The pipeline automatically generates a plot showing:
- The **smoothed abnormality score** over time.
- Dynamic statistical thresholds for `Moderate` (mean+2std), `Significant` (mean+3std), and `Severe` (mean+4std) anomalies.
- Color-coded dots highlighting detected anomalies.



## 🔧 Customization

You can easily customize the pipeline by modifying the core classes:

```python
# Customize the LSTM architecture
lstm_model = LSTMAutoencoder(timesteps=36, latent_dim=24)

# Choose your own ensemble of PyOD models
from pyod.models.abod import ABOD
from pyod.models.lof import LOF

pyod_models = [IForest(), ECOD(), LOF()]

# Adjust the ensemble weighting (LSTM, then PyOD models in order)
weights = [0.7, 0.1, 0.1, 0.1] # 70% weight to LSTM

detector = AnomalyDetector(lstm_model, pyod_models, weights)
```

## 📋 Dependencies

- **Python 3.7+**
- **pandas**: Data loading and manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Data scaling
- **tensorflow/keras**: LSTM Autoencoder model
- **pyod**: Ensemble outlier detection models
- **matplotlib/seaborn**: Visualization
