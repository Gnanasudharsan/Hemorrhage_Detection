import numpy as np
import cv2
from enum import Enum
from sklearn.decomposition import PCA

# Enum for selecting feature extraction method
class Method(Enum):
    SIMPLE = 1
    HISTOGRAM = 2
    HUMOMENTS = 3
    PCA = 4
    EDGES = 5

# First method: Resize the image to a fixed size and flatten it into a vector
def image_to_vector(image, size):
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()

# Second method: Compute histogram of the image and flatten it
def image_to_histogram_vector(image):
    histogram, bin_edges = np.histogram(image, bins=np.arange(257))
    histogram = np.reshape(histogram, (1, 256))
    return histogram

# Third method: Calculate the seven Hu moments (shape descriptors)
def fd_hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Fourth method: PCA for dimensionality reduction
def pca_reduction(images):
    pca = PCA(n_components=0.95)
    pca.fit(images)
    succinct_x = pca.transform(images)
    return succinct_x

# Fifth method: Canny edge detector and flatten the edges
def cany_edge(image, size):
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
    edges = cv2.Canny(image, 100, 200)
    edges = edges.flatten()
    return edges

# Main method to extract features from images
def extract_features(images, method=Method.SIMPLE, size=(32, 32)):
    succinct_x = []

    if method == Method.SIMPLE:
        flatten_size = size[0] * size[1]
        succinct_x = np.empty(shape=(0, flatten_size))
        for image in images:
            succinct_x = np.vstack([image_to_vector(image, size), succinct_x])

    elif method == Method.HISTOGRAM:
        succinct_x = np.empty(shape=(0, 256))
        for image in images:
            succinct_x = np.vstack([image_to_histogram_vector(image), succinct_x])

    elif method == Method.HUMOMENTS:
        succinct_x = np.empty(shape=(0, 7))
        for image in images:
            succinct_x = np.vstack([fd_hu_moments(image), succinct_x])

    elif method == Method.PCA:
        resized = [image_to_vector(image, size) for image in images]
        resized_images = np.stack(resized, axis=0)
        succinct_x = pca_reduction(resized_images)

    elif method == Method.EDGES:
        flatten_size = size[0] * size[1]
        succinct_x = np.empty(shape=(0, flatten_size))
        for image in images:
            succinct_x = np.vstack([cany_edge(image, size), succinct_x])

    return succinct_x

