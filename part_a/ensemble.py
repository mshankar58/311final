import numpy as np
from sklearn.impute import KNNImputer  # number of neighbors
from utils import *
from knn import knn_impute_by_user
from item_response import irt, sigmoid
from neural_network import train, AutoEncoder, load_data
from torch.autograd import Variable

# load data
sparse_matrix = load_train_sparse("../data").toarray()
train_data = load_train_csv("../data")
valid_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")
zero_train_matrix_nn, train_matrix_nn, valid_data_nn, test_data_nn = load_data("../data")

# knn
nbrs = KNNImputer(n_neighbors=11)
mat = nbrs.fit_transform(sparse_matrix)
pred_knn = []
for i in range(len(valid_data["user_id"])):
    cur_user_id = valid_data["user_id"][i]
    cur_question_id = valid_data["question_id"][i]
    pred_knn.append(mat[cur_user_id, cur_question_id])

# irt
theta, beta, val_acc_lst, NLL_train, NLL_validation = irt(train_data, valid_data, 0.005, 5)
pred_irt = []
for i, q in enumerate(valid_data["question_id"]):
    u = valid_data["user_id"][i]
    x = (theta[u] - beta[q]).sum()
    p_a = sigmoid(x)
    pred_irt.append(p_a)

# neural net
model = AutoEncoder(num_question=train_matrix_nn.shape[1], k=50)
train(model, 0.01, 0.001, train_matrix_nn,
      zero_train_matrix_nn, valid_data_nn, 5)  # returns list of accuracies over all epochs
pred_nn = []
for i, u in enumerate(valid_data_nn["user_id"]):
    inputs = Variable(train_matrix_nn[u]).unsqueeze(0)
    output = model(inputs)
    guess = output[0][valid_data_nn["question_id"][i]].item() >= 0.5
    pred_nn.append(guess)

# average the predictions
final_prob = np.add(pred_knn, pred_irt)
final_prob = np.array(final_prob) / 2

final_pred = []
for i in range(final_prob.size):
    if final_prob[i] >= 0.5:
        final_pred.append(1.)
    else:
        final_pred.append(0.)

final_pred = np.array(np.add(final_pred, pred_nn)) / 2
total_correct = 0
for i in range(len(final_pred)):
    if final_pred[i] == valid_data['is_correct'][i]:
        total_correct += 1
print(total_correct/len(final_pred))

# evaluate the redictions
# try comparing with 0.5 within each model then divide by 3 then compare 0.5 again
