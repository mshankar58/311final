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
    train_dic_0 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    train_matrix_0 = None
    zero_train_matrix_0 = None
    for n in range(len(train_data["user_id"])):
        i = np.random.randint(len(train_data["user_id"]))
        train_dic_0["user_id"].append(train_data["user_id"][i])
        train_dic_0["question_id"].append(train_data["question_id"][i])
        train_dic_0["is_correct"].append(train_data["is_correct"][i])
    for n in range(train_matrix_nn.shape[0]):
        i = np.random.randint(train_matrix_nn.shape[0])
        if train_matrix_0 is None:
            train_matrix_0 = train_matrix_nn[i]
            zero_train_matrix_0 = zero_train_matrix_nn[i]
        else:
            train_matrix_0 = torch.vstack((train_matrix_0, train_matrix_nn[i]))
            zero_train_matrix_0 = torch.vstack((zero_train_matrix_0, zero_train_matrix_nn[i]))
    train_dic_1 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    train_matrix_1 = None
    zero_train_matrix_1 = None
    for n in range(len(train_data["user_id"])):
        i = np.random.randint(len(train_data["user_id"]))
        train_dic_1["user_id"].append(train_data["user_id"][i])
        train_dic_1["question_id"].append(train_data["question_id"][i])
        train_dic_1["is_correct"].append(train_data["is_correct"][i])
    for n in range(train_matrix_nn.shape[0]):
        i = np.random.randint(train_matrix_nn.shape[0])
        if train_matrix_1 is None:
            train_matrix_1 = train_matrix_nn[i]
            zero_train_matrix_1 = zero_train_matrix_nn[i]
        else:
            train_matrix_1 = torch.vstack((train_matrix_1, train_matrix_nn[i]))
            zero_train_matrix_1 = torch.vstack((zero_train_matrix_1, zero_train_matrix_nn[i]))
    train_dic_2 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    train_matrix_2 = None
    zero_train_matrix_2 = None
    for n in range(len(train_data["user_id"])):
        i = np.random.randint(len(train_data["user_id"]))
        train_dic_2["user_id"].append(train_data["user_id"][i])
        train_dic_2["question_id"].append(train_data["question_id"][i])
        train_dic_2["is_correct"].append(train_data["is_correct"][i])

    for n in range(train_matrix_nn.shape[0]):
        i = np.random.randint(train_matrix_nn.shape[0])
        if train_matrix_2 is None:
            train_matrix_2 = train_matrix_nn[i]
            zero_train_matrix_2 = zero_train_matrix_nn[i]
        else:
            train_matrix_2 = torch.vstack((train_matrix_2, train_matrix_nn[i]))
            zero_train_matrix_2 = torch.vstack((zero_train_matrix_2, zero_train_matrix_nn[i]))
    # knn
    nbrs = KNNImputer(n_neighbors=11)
    mat = nbrs.fit_transform(train_matrix_0)
    pred_knn_0 = []
    for i in range(len(valid_data["user_id"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        pred_knn_0.append(mat[cur_user_id, cur_question_id])
    print("KNN 0 complete")
    mat = nbrs.fit_transform(train_matrix_1)
    pred_knn_1 = []
    for i in range(len(valid_data["user_id"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        pred_knn_1.append(mat[cur_user_id, cur_question_id])
    print("KNN 1 complete")
    mat = nbrs.fit_transform(train_matrix_1)
    pred_knn_2 = []
    for i in range(len(valid_data["user_id"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        pred_knn_2.append(mat[cur_user_id, cur_question_id])
    print("KNN 2 complete")

    # irt
    theta, beta, val_acc_lst, NLL_train, NLL_validation = irt(train_dic_0, valid_data, 0.005, 24)
    pred_irt_0 = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred_irt_0.append(p_a)
    print("IRT 0 complete")
    theta, beta, val_acc_lst, NLL_train, NLL_validation = irt(train_dic_1, valid_data, 0.005, 24)
    pred_irt_1 = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred_irt_1.append(p_a)
    print("IRT 1 complete")
    theta, beta, val_acc_lst, NLL_train, NLL_validation = irt(train_dic_2, valid_data, 0.005, 24)
    pred_irt_2 = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred_irt_2.append(p_a)
    print("IRT 2 complete")

    # neural net
    model = AutoEncoder(num_question=train_matrix_nn.shape[1], k=50)
    train(model, 0.01, 0.001, train_matrix_nn, zero_train_matrix_nn, valid_data_nn, 41)
    pred_nn = []
    for i, u in enumerate(valid_data_nn["user_id"]):
        inputs = Variable(zero_train_matrix_nn[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data_nn["question_id"][i]].item()
        pred_nn.append(guess)
    print("Neural Net 0 complete")
    # model = AutoEncoder(num_question=train_matrix_1.shape[1], k=50)
    # train(model, 0.01, 0.001, train_matrix_1, zero_train_matrix_1, valid_data_nn, 41)
    # pred_nn_1 = []
    # for i, u in enumerate(valid_data_nn["user_id"]):
    #     inputs = Variable(zero_train_matrix_1[u]).unsqueeze(0)
    #     output = model(inputs)
    #     guess = output[0][valid_data_nn["question_id"][i]].item()
    #     pred_nn_1.append(guess)
    # print("Neural Net 1 complete")
    # model = AutoEncoder(num_question=train_matrix_2.shape[1], k=50)
    # train(model, 0.01, 0.001, train_matrix_2, zero_train_matrix_2, valid_data_nn, 41)
    # pred_nn_2 = []
    # for i, u in enumerate(valid_data_nn["user_id"]):
    #     inputs = Variable(zero_train_matrix_2[u]).unsqueeze(0)
    #     output = model(inputs)
    #     guess = output[0][valid_data_nn["question_id"][i]].item()
    #     pred_nn_2.append(guess)
    # print("Neural Net 2 complete")

    # average the predictions
    pred_knn = np.add(np.add(pred_knn_0, pred_knn_1), pred_knn_2)
    pred_irt = np.add(np.add(pred_irt_0, pred_irt_1), pred_irt_2)
    # pred_nn = np.add(np.add(pred_nn_0, pred_nn_1), pred_nn_2)
    final_prob = np.add(np.add(pred_knn, pred_irt), pred_nn)
    final_prob = np.array(final_prob) / 7

    # evaluate the predictions
    final_pred = []
    for i in range(final_prob.size):
        if final_prob[i] >= 0.5:
            final_pred.append(1.)
        else:
            final_pred.append(0.)

    # tally totals
    total_correct = 0
    for i in range(len(final_pred)):
        if final_pred[i] == valid_data['is_correct'][i]:
            total_correct += 1
    print("FINAL ACCURACY:", total_correct / len(final_pred))


if __name__ == "__main__":
    main()
