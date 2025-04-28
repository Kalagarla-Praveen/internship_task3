

# 📄 Task 3: Linear Regression

## Objective
Implement and understand **Simple** and **Multiple Linear Regression** using:
- Scikit-learn
- Pandas
- Matplotlib

on a **Housing** dataset to predict house prices based on features like area, bedrooms, bathrooms, etc.

---

## Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Dataset
**Housing.csv**  
The dataset contains features such as:
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`
- `price` (target variable)

---

## Steps Performed
1. **Import Libraries**  
   Imported necessary libraries for data handling, visualization, and modeling.

2. **Load and Preprocess Dataset**  
   - Loaded the `Housing.csv` file.
   - Selected features for Simple and Multiple Linear Regression.

3. **Train-Test Split**  
   - Divided data into training (80%) and testing (20%) sets.

4. **Model Building**  
   - Used `LinearRegression` from `sklearn.linear_model`.
   - Fitted the model on training data.

5. **Model Evaluation**  
   Evaluated the model using:
   - **MAE** (Mean Absolute Error)
   - **MSE** (Mean Squared Error)
   - **R² Score** (Coefficient of Determination)

6. **Visualization**  
   - Plotted the regression line for simple linear regression (Area vs Price).

7. **Interpretation**  
   - Displayed and interpreted the model coefficients (Intercept and Slope).

---

## How to Run

1. Clone the repository or download the code files.
2. Make sure you have Python installed (preferably 3.8+).
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

4. Run the script:

```bash
python linear_regression_housing.py
```

---

## Results
- Trained a regression model to predict house prices.
- Achieved a good **R² Score** showing how well the model fits the data.
- Visualized the regression line for better understanding.
- Interpreted how each feature affects the house price.

---

## Conclusion
Linear Regression is a foundational supervised learning technique.  
By using the Housing dataset:
- We understood how a single or multiple independent features can influence a dependent variable.
- Learned how to evaluate model performance.
- Visualized results to interpret the model easily.

---

