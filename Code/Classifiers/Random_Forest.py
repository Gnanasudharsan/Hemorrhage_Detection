from sklearn.ensemble import RandomForestClassifier
import Code.Draw

def create_random_forest_model(n_estimators=1000, random_state=None, max_depth=None):
    """
    Create a RandomForest model with the specified parameters.
    """
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)

def random_forest_classification(train_data_X, train_data_Y, test_data_X, test_data_Y, img_collection, img_index):
    # Create the RandomForest model
    rf_model = create_random_forest_model(n_estimators=1000, random_state=42)
    
    # Train the model
    rf_model.fit(train_data_X, train_data_Y)
    
    # Calculate the accuracy
    accuracy_percent = rf_model.score(test_data_X, test_data_Y) * 100
    
    # Visualize predictions
    Code.Draw.drawPredict(rf_model, test_data_X, test_data_Y, img_collection, img_index)
    
    return accuracy_percent

