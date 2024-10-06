from sklearn.tree import DecisionTreeClassifier
import Code.Draw

def decision_tree_model(train_X, train_y, test_X, test_y, image_set, idx):
    tree_model = DecisionTreeClassifier(random_state=42)  # Added random state for consistency
    tree_model.fit(train_X, train_y)
    accuracy = tree_model.score(test_X, test_y) * 100
    Code.Draw.drawPredict(tree_model, test_X, test_y, image_set, idx)
    return accuracy
