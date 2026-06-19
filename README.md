# 🛡️ ThreatScope: Insider Threat Detection Dashboard

ThreatScope is a machine learning-powered security analytics dashboard designed to detect potential insider threats using organizational email metadata. By training a robust Random Forest classifier, the system surfaces anomalous behavior patterns (such as data exfiltration signals, weekend work hours, and bulk data transfers) in real time.

The application features a premium, professional minimalist black-and-white interface built with Streamlit.

---

## 🚀 Key Features & Dashboard Sections

### 1. Overview & Behavioral Signals
The dashboard presents high-level statistics of analyzed emails, unique users, and flagged threat rates. The classifier utilizes **four primary behavioral signals**:
* **📧 Email Volume (`email_count`)**: Total emails sent per user. A sudden surge in volume acts as a primary indicator of anomalous behavior.
* **📎 Attachments (`attachments`)**: Total email attachments. Data exfiltration often correlates with a high frequency of attachments.
* **📅 Day of Week (`day_of_week`)**: Activity during weekends (Saturday–Sunday) is flag-triggering in standard corporate environments.
* **📦 Email Size (`size`)**: Large emails point to bulk data transfers and potential unauthorized archiving.

### 2. Hyperparameter Tuning (Grid Search CV)
A clean, interactive tuning panel lets security teams define search grids for the Random Forest model:
* **Tree Structure**: Customize tree count (`n_estimators`) and depth limits (`max_depth`).
* **Leaf & Split Criteria**: Tune node splits (`min_samples_split`) and leaf samples (`min_samples_leaf`).
* **Cross-Validation**: Runs 5-fold cross-validated grid search automatically.
* **Tuning Metrics**: Showcases test set delta improvements (**Before vs. After Tuning**) along with the optimal parameter configurations and updated feature importance bars.

### 3. Model Evaluation & Justification
Comprehensive analysis of model performance using standard ML metrics on a 20% held-out test split:
* **Metrics Reported**: Accuracy, Precision, Recall, F1 Score, and AUC-ROC.
* **Explainable Confusion Matrix**: Provides absolute counts for True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN) paired with security context definitions.
* **Algorithm Justification**: Outlines six evidence-backed reasons (Ensemble Stability, Imbalance Handling, Native Feature Importance, Non-Linear Interactions, Scale Invariance, and Calibrated Risk Scores) why Random Forest is chosen over Logistic Regression, SVM, and Single Decision Trees.
* **Benchmark Table**: Interactive comparative comparison displaying accuracy, precision, recall, and F1 columns side-by-side for all evaluated algorithms.

### 4. Live Risk Assessment
A real-time inference playground allowing security analysts to manually input behavioral statistics and receive instantaneous threat probability verdicts:
* **Verdict Indicators**: Visual alerts indicating whether the user behavior pattern is a **Potential Insider Threat** (in high-contrast danger red) or **No Threat Detected**.
* **Risk Score**: A modern confidence bar depicting probability levels (0-100%).
* **Indicator Breakdown**: A row-by-row parameter breakdown highlighting exactly which thresholds were crossed (e.g., volume > 30 emails, size > 1MB, weekend work).

---

## 📊 Ground Truth Threat Heuristics

A record is labeled as a potential insider threat if any of the following heuristic rules are met:
1. **Email Count > 30**: Extreme sending volume.
2. **Attachments > 5**: High attachment density.
3. **Day of Week ≥ 4**: Weekend usage (Saturday & Sunday).
4. **Email Size > 1,000,000 bytes (1MB)**: Bulk data transfer.

---

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shreeya17n-a11y/Insider-Threat-Detection.git
   cd Insider-Threat-Detection
   ```
2. Install the required libraries:
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 🎨 Visual Design
* **Minimalist Aesthetics**: Designed with a clean, high-contrast black-and-white UI for professional enterprise deployment.
* **Centered Grid Layout**: Optimized container alignment constraining the app width to `1100px` for enhanced readability on wide screens.
* **Horizontal Tab-Bar Navigation**: Intuitive tab navigation with dynamic hover states.
