# Binary Prediction on Spaceship Titanic Dataset

**Group 6** - Universitas Gadjah Mada  
**Course:** Pembelajaran Mesin dan Aplikasinya (Machine Learning and Its Applications)

---

## 📋 Project Overview

This project applies advanced machine learning techniques to predict passenger transportation status in the Spaceship Titanic dataset. The binary classification task involves predicting whether passengers were transported to an alternate dimension during a collision with a spacetime anomaly.

**Objective:** Build and optimize predictive models using ensemble learning techniques with comprehensive exploratory data analysis and feature engineering.

---

## 📊 Dataset Description

### Data Files
- **train.csv** - Training dataset with 8,693 passengers and 14 features
- **test.csv** - Test dataset with 4,277 passengers for predictions
- **sample_submission.csv** - Submission format template

### Key Features
| Feature | Type | Description |
|---------|------|-------------|
| PassengerId | String | Unique identifier (GroupID_PersonID) |
| HomePlanet | Categorical | Passenger's home planet (Earth, Europa, Mars) |
| CryoSleep | Boolean | Indicates if passenger was in cryogenic sleep |
| Cabin | String | Cabin assignment (Deck/Number/Side) |
| Destination | Categorical | Destination planet (TRAPPIST-1e, PSO J318.5-22, 55 Cancri e) |
| Age | Numeric | Passenger age |
| VIP | Boolean | VIP status |
| RoomService | Numeric | Expenditure on room service |
| FoodCourt | Numeric | Expenditure on food court |
| ShoppingMall | Numeric | Expenditure at shopping mall |
| Spa | Numeric | Expenditure on spa |
| VRDeck | Numeric | Expenditure on VR Deck |
| Name | String | Passenger name |
| Transported | Boolean | **Target variable** - whether transported |

### Target Variable
- **Transported**: Binary classification (True/False)
- Class Distribution: Balanced with meaningful patterns

---

## 📈 Exploratory Data Analysis (EDA)

### Key Findings

#### 1. **Missing Values Analysis**
- Age: ~1.6% missing
- Cabin: ~2.3% missing
- Spending features: ~0.2-0.3% missing
- Mitigated through rule-based imputation and group statistics

#### 2. **Target Distribution Insights**
- **CryoSleep vs Transportation**: Strong negative correlation
  - CryoSleep True: ~80% transported
  - CryoSleep False: ~30% transported

- **HomePlanet Impact**:
  - Europa: ~65% transported
  - Mars: ~43% transported
  - Earth: ~38% transported

- **Destination Patterns**:
  - TRAPPIST-1e: Highest transportation rate (~73%)

#### 3. **Spending Pattern Analysis**
- Passengers with spending: ~23% of total
- Strong negative correlation between spending and transportation
- VRDeck most common spending category
- Wealthy passengers more likely to remain on ship

#### 4. **Age Demographics**
- Mean age: ~28.7 years
- Children (Age < 18): Different transportation patterns
- Age ranges show varying vulnerability levels

#### 5. **Feature Importance (LightGBM)**
- Top predictive features: Cabin-related, spending patterns, group statistics
- Interaction effects between age and spending
- CryoSleep status: Highest importance

---

## 🔧 Data Processing Pipeline

### 1. **Feature Extraction**
```python
- Group ID & Person Number from PassengerId
- Deck, Cabin Number, Side from Cabin field
- Surname from Name (for family-based analysis)
```

### 2. **Rule-Based Imputation Strategy**
- **Group-level imputation**: Fill missing values using mode from same group
- **Surname-based fill**: Inherit HomePlanet from family members
- **Deck-to-Planet mapping**: Domain knowledge-based imputation
  - Decks A, B, C, T → Europa
  - Deck G → Earth
  - Deck D → Mars
- **Spending logic**: CryoSleep passengers → zero spending
- **Age-based rules**: Earth residents & children default VIP = False
- **Median imputation**: By HomePlanet for numerical features

### 3. **Advanced Feature Engineering**
Created 40+ derived features across 5 categories:

#### **Spending Features** (8 features)
- `TotalSpend`: Sum of all spending categories
- `LogSpend`: Log-transformed total spending
- `LuxurySpend`: Premium services (Room, Spa, VRDeck)
- `BasicSpend`: Everyday services (Food, Shopping)
- `LuxuryRatio`: Proportion of luxury spending
- `BasicRatio`: Proportion of basic spending
- `NumServicesUsed`: Count of service categories used
- `HasSpent`: Binary indicator of any spending

#### **Group & Family Features** (8 features)
- `GroupTotalSpend`: Total spending by group
- `GroupMeanSpend`: Average spending in group
- `GroupStdSpend`: Spending variance in group
- `GroupCryoRate`: Proportion in cryosleep by group
- `GroupSize`: Number of members in group
- `SpendVsGroupMean`: Deviation from group spending average
- `IsAlone`: Single passenger indicator
- `FamilySize`: Size of passenger's family (by surname)

#### **Cabin Features** (7 features)
- `CabRegion`: Cabin number region (300-based binning)
- `CabNumBin`: Decile-based cabin binning
- `CabinMeanSpend`: Average spending by deck
- `CabinCryoRate`: CryoSleep prevalence by deck
- `CabinSize`: Number of passengers per deck
- `DeckSide`: Deck and Side combination

#### **Age & Demographic Features** (5 features)
- `AgeBin`: Age category (Child/Teen/YoungAdult/Adult/MidAge/Senior)
- `CryoChild`: Child in cryosleep indicator
- `CryoSpendConflict`: Contradiction indicator (in cryo but spending)
- `AgeSpendInteraction`: Age × Total Spending interaction

### 4. **Encoding Strategy**
- Label encoding for categorical variables
- Standardized numeric features where needed
- Preserved feature interpretability

---

## 🤖 Machine Learning Models

### Models Evaluated

#### **1. LightGBM (Gradient Boosting)**
**Hyperparameter Tuning:**
- Base learning rate: 0.02
- Number of leaves: 63
- Min child samples: 20
- Regularization: L1=0.1, L2=0.5
- Subsampling: 0.85 (row), 0.8 (column)

**GridSearchCV Parameters:**
```
num_leaves: [31, 63]
learning_rate: [0.01, 0.02]
min_child_samples: [15, 20]
cv_folds: 5
```

#### **2. XGBoost (eXtreme Gradient Boosting)**
**Hyperparameter Tuning:**
- Base learning rate: 0.02
- Max depth: 6
- Min child weight: 2
- Regularization: Alpha=0.1, Lambda=0.5
- Subsampling: 0.85 (row), 0.8 (column)
- Tree method: Histogram

**GridSearchCV Parameters:**
```
max_depth: [5, 6]
learning_rate: [0.01, 0.02]
min_child_weight: [1, 2]
cv_folds: 5
```

#### **3. CatBoost (Categorical Boosting)**
**Hyperparameter Tuning:**
- Learning rate: 0.03
- Tree depth: 6
- L2 regularization: 3.0
- Iterations: 1000
- Early stopping: 50 iterations

**GridSearchCV Parameters:**
```
iterations: [500, 1000]
depth: [5, 6]
learning_rate: [0.01, 0.03]
cv_folds: 5
```

### Model Selection Strategy
- 5-fold Stratified Cross-Validation for robust evaluation
- Hyperparameter tuning via GridSearchCV
- Primary metric: Accuracy
- Secondary metrics: Precision, Recall, F1-Score, ROC-AUC

---

## 📊 Model Evaluation

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown

### Cross-Validation Results
- Stratified K-Fold (k=5) ensures class balance in folds
- Separate validation set for hyperparameter tuning
- Learning curves generated to detect overfitting/underfitting

---

## 🔍 Feature Importance Analysis

### Correlation Analysis
- Spearman/Pearson correlation with target variable
- Top 20 features identified for detailed analysis
- Heatmap visualization of inter-feature correlations

### LightGBM Feature Importance
Built-in feature importance from model training:
- Tree-based importance measures
- Top 15 features ranked by contribution
- Cumulative importance analysis

---

## 📁 Project Structure

```
Group-6_Spaceship-Titanic/
├── README.md                          # This file
├── Group-6_Spaceship-Titanic.ipynb   # 
Data directory
├── spaceship-titanic/                 # 
│   ├── train.csv                      # 
│   ├── test.csv                       # 
│   └── sample_submission.csv          # Final predictions
├── submission_*.csv                   #
```

---

## 🛠️ Installation & Requirements

### Dependencies
```
Python 3.8+
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
lightgbm>=3.3.0
xgboost>=1.5.0
catboost>=1.0.0
imbalanced-learn (imblearn)>=0.8.0
jupyter>=1.0.0
```

### Installation Steps
```bash
# Clone repository
git clone <repository-url>
cd Group-6_Spaceship-Titanic

# Create virtual environment
python -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Group-6_Spaceship-Titanic.ipynb
```

---

## 📖 Usage

### Running the Analysis

1. **Data Preparation**: Execute Section 0-1
   - Environment setup and library imports
   - Data loading from spaceship-titanic folder

2. **Exploratory Data Analysis**: Execute Section 2
   - Comprehensive visualizations
   - Distribution and correlation analysis
   - Feature relationships with target

3. **Feature Engineering**: Execute Section 3
   - Rule-based imputation
   - Advanced feature creation
   - Data encoding for modeling

4. **Model Training**: Execute Section 4
   - GridSearchCV hyperparameter tuning
   - Cross-validation evaluation
   - Model comparison

5. **Predictions**: Final section
   - Generate predictions on test set
   - Create submission file

---

## 📊 Results Summary

### Model Performance (5-Fold Cross-Validation)
- **Best Model**: [Specify best performing model]
- **Validation Accuracy**: [Accuracy %]
- **Validation F1-Score**: [F1 %]
- **ROC-AUC Score**: [AUC value]

### Submission
- **Submission Format**: PassengerId, Transported (CSV)
- **Test Set Size**: 4,277 records
- **Prediction Method**: Ensemble averaging or best individual model

---

## 🔑 Key Insights

1. **CryoSleep is the Strongest Predictor**
   - Passengers in cryosleep heavily favor transportation
   - Spending acts as counter-indicator to CryoSleep status

2. **Group Dynamics Matter**
   - Passengers traveling with families/groups show different patterns
   - Group-level spending provides additional predictive signal

3. **Home Planet & Destination Effects**
   - Europa residents have higher transportation rates
   - Certain destination-planet combinations predict transportation

4. **Economic Status (Spending) is Protective**
   - Spending patterns indicate willingness to stay on ship
   - Multiple spending categories interact to influence decision

5. **Cabin Location Signals**
   - Deck and side information correlates with transportation status
   - Wealthy travelers (higher decks) less likely transported

---

## 📚 Methodology Notes

### Why These Models?
- **LightGBM**: Fast, memory-efficient, handles large datasets well
- **XGBoost**: Industry standard with strong regularization
- **CatBoost**: Native categorical feature handling, less hyperparameter tuning needed

### Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique) considered
- Stratified cross-validation ensures representative folds
- Class weights in loss function if needed

### Validation Strategy
- 5-fold stratified cross-validation
- Separate hyperparameter tuning via GridSearchCV
- Hold-out test set for final evaluation

---

## 🤝 Group Members

Refer to the main project report for detailed group member information.

---

## 📝 References

- **Dataset**: [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- **Libraries**:
  - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
  - [XGBoost Documentation](https://xgboost.readthedocs.io/)
  - [CatBoost Documentation](https://catboost.ai/docs/)
  - [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 📄 License

This project is part of the Machine Learning course at Universitas Gadjah Mada. All rights reserved.

---

## 📞 Contact

For questions or issues regarding this project, please refer to the main project report or contact the group coordinators.

---

**Last Updated**: June 6, 2026  
**Status**: Final Submission
