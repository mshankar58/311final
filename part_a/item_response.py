import random

from matplotlib import pyplot as plt

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
    is_correct = np.array(data['is_correct'])
    l = is_correct*np.log(sig) + ((1-is_correct)*np.log(1-sig))
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
    number_of_samples = len(data["user_id"])

    theta_exp = np.exp(theta)
    beta_exp = np.exp(beta)

    for sample_index in range(number_of_samples):
        user_id = data["user_id"][sample_index]
        question_id = data["question_id"][sample_index]
        c_ij = data["is_correct"][sample_index]

        b_exp = beta_exp[question_id] / (beta_exp[question_id] + theta_exp[user_id])
        th_exp = theta_exp[user_id] / (beta_exp[question_id] + theta_exp[user_id])

        grad_theta[user_id] += c_ij * b_exp - (1 - c_ij) * th_exp
        grad_beta[question_id] += (1 - c_ij) * th_exp - c_ij * b_exp

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
    theta = np.random.uniform(low=0.0, high=1.0, size=len(set(data['user_id'])))
    beta = np.random.uniform(low=0.0, high=1.0, size=len(set(data['question_id'])))
    val_acc_lst = []
    neg_log_likeli_train = []
    neg_log_likeli_validation = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_log_likeli_train.append(neg_lld_train)

        neg_lld_validation = neg_log_likelihood(val_data, theta=theta, beta=beta)
        neg_log_likeli_validation.append(neg_lld_validation)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)

        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_log_likeli_train, neg_log_likeli_validation


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
    theta, beta, acc_list, neg_log_likeli_list_train, neg_log_likeli_list_validation =irt(train_data, val_data, 0.005, 70)
    plt.figure(1)
    plt.plot(acc_list)
    plt.title("Accuracy over iterations")
    plt.figure(2)
    plt.plot(neg_log_likeli_list_train)
    plt.title("Negative likelihood over iterations on training set")
    plt.figure(3)
    plt.plot(neg_log_likeli_list_validation)
    plt.title("Negative likelihood over iterations on validation set")
    print("Accuracy on Test set", evaluate(test_data, theta, beta))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plt.figure(4)
    # j1, j2, j3
    for i in range(3):
        probability = []
        thetas = []
        for j in theta:
            # p(c_ij = 1) = sigmoid
            probability.append(sigmoid(j - beta[i]))
            thetas.append(j)
        thetas.sort()
        probability.sort()
        plt.plot(thetas, probability)
    plt.gca().legend(('j1','j2','j3'))
    plt.title("Probability of The correct response over Theta")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
