from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print("Sparse matrix:")
    # print(sparse_matrix)
    # print("Shape of sparse matrix:")
    # print(sparse_matrix.shape)
    print(test_data.keys())
    print(len(test_data['question_id']))
    print(len(test_data['is_correct']))

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # acc_by_user = []
    # ks = [1,6,11,16,21,26]
    # for i in ks:
    #     acc_by_item.append(knn_impute_by_user(sparse_matrix, valid_data=val_data, k=i))
    # plt.plot(ks, acc_by_user)
    # plt.show()

    # acc_by_item = []
    # ks = [1,6,11,16,21,26]
    # for i in ks:
    #     acc_by_item.append(knn_impute_by_item(sparse_matrix, valid_data=val_data, k=i))
    # plt.plot(ks, acc_by_item)
    # plt.show()

    knn_impute_by_user(sparse_matrix,test_data, 11)
    knn_impute_by_item(sparse_matrix,test_data, 21)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
