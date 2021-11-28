import numpy as np
import torch
from sklearn.impute import KNNImputer  # number of neighbors
from utils import *
from item_response import irt, sigmoid
from neural_network import train, AutoEncoder, load_data
from torch.autograd import Variable


def main():
    # load data
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    zero_train_matrix_nn, train_matrix_nn, valid_data_nn, test_data_nn = load_data("../data")

    # split the train data into 3 datasets with size = train data
    number_of_bags = 3

    train_dic = []
    train_matrix = []
    zero_train_matrix = []
    for bagging_index in range(number_of_bags):
        train_dic.append({
            "user_id": [],
            "question_id": [],
            "is_correct": []
        })
        dic_bag = train_dic[bagging_index]

        train_matrix.append(torch.empty(0, train_matrix_nn.shape[1]))
        zero_train_matrix.append(torch.empty(0, zero_train_matrix_nn.shape[1]))

        for n in range(len(train_data["user_id"])):
            random_index = np.random.randint(len(train_data["user_id"]))
            dic_bag["user_id"].append(train_data["user_id"][random_index])
            dic_bag["question_id"].append(train_data["question_id"][random_index])
            dic_bag["is_correct"].append(train_data["is_correct"][random_index])

        for n in range(train_matrix_nn.shape[0]):
            random_index = np.random.randint(train_matrix_nn.shape[0])
            train_matrix[bagging_index] = torch.vstack((train_matrix[bagging_index], train_matrix_nn[random_index]))
            zero_train_matrix[bagging_index] = torch.vstack((zero_train_matrix[bagging_index], zero_train_matrix_nn[random_index]))

    # train models
    pred_knn = []
    pred_irt = []
    pred_nn = []
    for i in range(number_of_bags):
        # knn
        nbrs = KNNImputer(n_neighbors=11)
        mat = nbrs.fit_transform(train_matrix[i])
        acc = sparse_matrix_evaluate(valid_data, mat)
        print("Validation Accuracy: {}".format(acc))
        pred_knn.append([])
        predicted_knn_correctness = pred_knn[i]

        for student in range(len(valid_data["user_id"])):
            cur_user_id = valid_data["user_id"][student]
            cur_question_id = valid_data["question_id"][student]
            predicted_knn_correctness.append(mat[cur_user_id, cur_question_id])
        print("KNN " + str(i) + " complete")


        # irt
        theta, beta, val_acc_lst, NLL_train, NLL_validation = irt(train_dic[i], valid_data, 0.005, 24)
        pred_irt.append([])
        predicted_irt_correctness = pred_knn[i]
        for student, q in enumerate(valid_data["question_id"]):
            u = valid_data["user_id"][student]
            x = (theta[u] - beta[q]).sum()
            p_a = sigmoid(x)
            predicted_irt_correctness.append(p_a)
        print("IRT " + str(i) + " complete")


        # neural net
        pred_nn.append([])
        predicted_nn_correctness = pred_nn[i]
        model = AutoEncoder(num_question=train_matrix[i].shape[1], k=50)
        train(model, 0.01, 0.001, train_matrix[i], zero_train_matrix[i], valid_data_nn, 20)
        for j, u in enumerate(valid_data_nn["user_id"]):
            inputs = Variable(zero_train_matrix[i][u]).unsqueeze(0)
            output = model(inputs)
            guess = output[0][valid_data_nn["question_id"][j]].item()
            predicted_nn_correctness.append(guess)
        print("Neural Net " + str(i) + " complete")

    # average the predictions
    average_pred_knn = pred_knn[0]
    average_pred_irt = pred_irt[0]
    average_pred_nn = pred_nn[0]
    for i in range(1, number_of_bags):
        average_pred_knn = np.add(average_pred_knn, pred_knn[i])
        average_pred_irt = np.add(average_pred_irt, pred_irt[i])
        average_pred_nn = np.add(average_pred_nn, pred_nn[i])

    final_prob = np.add(np.add(pred_knn, pred_irt), pred_nn)
    final_prob = np.array(final_prob) / (3 * number_of_bags)

    # evaluate the predictions
    final_pred = []
    for bagging_index in range(final_prob.size):
        if final_prob[bagging_index] >= 0.5:
            final_pred.append(1.)
        else:
            final_pred.append(0.)

    # tally totals
    total_correct = 0
    for bagging_index in range(len(final_pred)):
        if final_pred[bagging_index] == valid_data['is_correct'][bagging_index]:
            total_correct += 1
    print("FINAL ACCURACY:", total_correct / len(final_pred))


if __name__ == "__main__":
    main()
