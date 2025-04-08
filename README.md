# 🇫🇷 Predicting Restaurant Success in France (TripAdvisor Dataset)

This project aims to predict whether a restaurant in **France** will be successful based on a variety of available features (tags, cuisine, location, operating hours, awards, etc.).

---

## 📦 Dataset

- **Source**: [TripAdvisor European Restaurants – Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants)
- **Scope**: Focused on restaurants located in **France**.
- **Google Drive (Raw, Processed & Outputs)**:  
  [📁 View Full Project Data and Outputs](https://drive.google.com/drive/folders/1lbXOX9aFPgftf4-BWSRJHOaNiXcqz_qO?usp=sharing)

---

## 📊 Project Summary

1. **Goal**: To test whether a given restaurant in France is likely to be **successful**.
2. **Success Criteria**:  
   - Defined using the **75th percentile** of both:
     - Number of reviews
     - Average review score
   - This creates a balanced label for classification purposes.
3. **Review Data Handling**:  
   - All review-related features are saved in `dropped_columns.csv` for future reference.  
   - These features were **excluded from training** because review data is **not available for new restaurants**.
   - The dropped review columns can be used later to refine the success criteria
4. **Awards Consideration**:  
   - If a restaurant **has moved** and held awards previously, that is preserved.  
   - For new locations or new restaurants, award data is **set to 0 or unknown**.

---

## 🧠 Code Structure

The project is organized into two main directories for code and experimentation:

### 🗂️ `src/` — Utility Functions
This folder contains Python modules with **reusable functions** developed for the project, such as:
- Data preprocessing
- Feature encoding
- Data splitting (`prepare_data`)
- Utility logic used across notebooks

> All key logic is modularized here to keep the notebooks clean and focused.

### 📓 `notebooks/` — Main Project Work
All core modeling, analysis, and visualizations were performed inside Jupyter notebooks:
- 📊 Exploratory analysis
- ⚙️ Model training
- 🔍 Feature selection
- 📈 Performance comparisons
- 💾 Model saving and outputs

These notebooks are well-commented and structured to reflect the full pipeline from raw data to final model evaluation.

---

## 📁 Folder Structure (Google Drive)

- `/data/raw_data` – Original Kaggle data
- `/data/processed_data` – Cleaned and encoded feature sets
- `/data/external` – Any external data used
- `/outputs/results` – Feature selection summaries, evaluation metrics
- `/outputs/figures` – Plots and visualizations
- `/outputs/models` – Final trained models (`.pkl`) ready to use

---

## 🛠 Known Issues

- ❌ A compatibility issue was encountered with **XGBoost and Scikit-learn version ≥1.6**, which led to errors related to `__sklearn_tags__`.
- ✅ This was resolved by **manually running the hyperparameter grid** (instead of using `RandomizedSearchCV`), allowing for stable and controlled optimization.

---

## 🚀 Future Plans

1. 🔧 **Interactive Input UI**  
   - A notebook or web interface where users can input their restaurant’s data and receive a prediction.
2. 🌍 **Generalization to Other Countries**  
   - Extend the model to work across Europe, and eventually globally, by retraining on a more diverse dataset.

---

## 📁 Folder Structure (Google Drive)

- `/data/raw_data` – Original Kaggle data
- `/data/processed_data` – Cleaned and encoded feature sets
- `/data/external` – any external data used
- `/outputs/results` – Feature selection summaries, evaluation metrics
- `/outputs/figures` – Plots and visualizations
- `/outputs/models` – Final trained models (`.pkl`) ready to use

---

## 📦 Installation

You can set up the project environment using either **Poetry** or `requirements.txt`.

### ▶️ Option 1: Using Poetry (recommended)

1. Clone the repository or download the code
2. Navigate to the project folder
3. Run:

```bash
poetry install
poetry shell
```

This will automatically install the correct Python version and all dependencies from the `pyproject.toml`.

> Make sure you have [Poetry installed](https://python-poetry.org/docs/#installation).

---

### ▶️ Option 2: Using `requirements.txt`

If you prefer to use `pip`:

```bash
pip install -r requirements.txt
```

This installs all required packages based on the project's environment.

---

## 💡 Final Notes

This project offers a strong starting point for restaurant business analysis in France, with potential for real-world applications in business planning, location scouting, and investor forecasting.