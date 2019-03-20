import numpy as np
from q2e_word2vec import normalizeRows


def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []

    ### YOUR CODE
    ### END YOUR CODE
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
        your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE

    indices = knn(np.array([0.2,0.5]), np.array([[0,0.5],[0.1,0.1],[0,0.5],[2,2],[4,4],[3,3]]), k=2)
    assert 0 in indices and 2 in indices and len(indices) == 2

    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


