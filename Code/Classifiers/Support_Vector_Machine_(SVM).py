from sklearn.svm import SVC
import Code.Draw

class SVMModel:
    def __init__(self, kernel_type='linear'):
        self.model = SVC(kernel=kernel_type)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test, images, testIm):
        accuracy = self.model.score(X_test, y_test) * 100
        Code.Draw.drawPredict(self.model, X_test, y_test, images, testIm)
        return accuracy

def svm_kernel_classification(X_train, y_train, X_test, y_test, images, testIm, kernel_type='linear'):
    svm_model = SVMModel(kernel_type)
    svm_model.train(X_train, y_train)
    return svm_model.evaluate(X_test, y_test, images, testIm)

