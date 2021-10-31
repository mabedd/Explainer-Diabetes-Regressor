# Importing Libraries & Packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from sklearn.datasets import load_diabetes

"""Build and Train a Regressor"""
# Import datasets
data = load_diabetes()

# Create DF from the dataset features
X = pd.DataFrame(data.data, columns=data.feature_names)

# Create DF from the dataset target
y = pd.DataFrame(data.target, columns=['target'])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Train the model
model = RandomForestRegressor(n_estimators=50, max_depth=5)
model.fit(X_train, y_train.values.ravel())


"""Setup the Explainable Dashboard"""
# Initialize the explainer
explainer = RegressionExplainer(model, X_test, y_test)

# Start the dashboard
db = ExplainerDashboard(explainer, title="Diabetes Predictions", whatif=False)

# Run the app
db.run()
