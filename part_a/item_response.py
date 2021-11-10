from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    sig = sigmoid(np.array(theta[data['user_id']]) - np.array(beta[data['question_id']]))
    l = np.array(data['is_correct'])*np.log(sig) + ((1-np.array((data['is_correct'])))*np.log(1-sig))
    log_lklihood = l.sum()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    grad_theta = np.zeros(theta.shape)
    grad_beta = np.zeros(beta.shape)
    num = len(data["user_id"])
    for k in range(num):
        i = data["user_id"][k]
        j = data["question_id"][k]
        c_ij = data["is_correct"][k]
        b_exp = np.exp(beta)[j] / (np.exp(beta)[j] + np.exp(theta)[i])
        th_exp = np.exp(theta)[i] / (np.exp(beta)[j] + np.exp(theta)[i])
        grad_theta[i] += ((c_ij * b_exp) - ((1 - c_ij) * th_exp))
        grad_beta[j] += ((c_ij * (-b_exp)) + ((1 - c_ij) * th_exp))
    theta += lr * grad_theta
    beta += lr * grad_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    theta = np.random.uniform(low=0.0, high=1.0, size=len(data['user_id']))
    beta = np.random.uniform(low=0.0, high=1.0, size=len(data['question_id']))
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # print(sparse_matrix.shape)
    # p = [x for x in range(len(train_data['user_id'])) if train_data['user_id'][x] == 488]
    # q = [train_data['question_id'][y] for y in p]
    # r = [train_data['is_correct'][y] for y in p]
    # print("user id: 488")
    # print(q)
    # print(r)
    print("")
    print(irt(train_data, val_data, 0.005, 20)[2])
    print(irt(train_data, val_data, 0.01, 20)[2])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
