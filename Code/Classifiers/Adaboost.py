from sklearn.ensemble import AdaBoostClassifier
import Code.Draw

def adaBoostClassifierModel(train_X, train_y, test_X, test_y, img_data, idx):
    ada_boost_model = AdaBoostClassifier()
    # Fit the model with training data
    ada_boost_model.fit(train_X, train_y)
    accuracy = ada_boost_model.score(test_X, test_y) * 100
    Code.Draw.drawPredict(ada_boost_model, test_X, test_y, img_data, idx)
    return accuracy
