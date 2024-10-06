from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wasserstein_distance
import Code.Draw

def create_knn_model(neighbors=3, metric='minkowski', algorithm='auto'):
    """Creates and returns a KNeighborsClassifier based on the specified metric and algorithm."""
    return KNeighborsClassifier(n_neighbors=neighbors, metric=metric, algorithm=algorithm, n_jobs=-1)

def knn_classification(train_X, train_Y, test_X, test_Y, images, idx, neighbors=3):
    knn_model = create_knn_model(neighbors)
    knn_model.fit(train_X, train_Y)
    accuracy = knn_model.score(test_X, test_Y) * 100
    Code.Draw.drawPredict(knn_model, test_X, test_Y, images, idx)
    return accuracy

def knn_emd_classification(train_X, train_Y, test_X, test_Y, images, idx, neighbors=3):
    knn_emd_model = create_knn_model(neighbors, metric=EMD, algorithm='ball_tree')
    knn_emd_model.fit(train_X, train_Y)
    accuracy_emd = knn_emd_model.score(test_X, test_Y) * 100
    Code.Draw.drawPredict(knn_emd_model, test_X, test_Y, images, idx)
    return accuracy_emd

def EMD(x, y):
    return wasserstein_distance(x, y)

