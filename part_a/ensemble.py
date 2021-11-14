from sklearn.impute import KNNImputer  # number of neighbors
from utils import *
from knn import knn_impute_by_user
from item_response import irt, sigmoid
from neural_network import train, AutoEncoder, load_data

# load data
sparse_matrix = load_train_sparse("../data").toarray()
train_data = load_train_csv()
valid_data = load_valid_csv()
test_data = load_public_test_csv()
zero_train_matrix_nn, train_matrix_nn, valid_data_nn, test_data_nn = load_data("../data")

# knn
nbrs = KNNImputer(n_neighbors=21)
mat = nbrs.fit_transform(sparse_matrix)
# # TODO: get predictions (look at utils.sparse_matrix_evaluate)

#irt
theta, beta, val_acc_lst, NLL_train, NLL_validation = irt(train_data, valid_data, 0.005, 70)
pred = []
for i, q in enumerate(train_data["question_id"]):
      u = train_data["user_id"][i]
      x = (theta[u] - beta[q]).sum()
      p_a = sigmoid(x)
      pred.append(p_a >= 0.5)

# neural net
train(AutoEncoder(num_question=train_matrix_nn.shape[1], k=50), 0.01, 0.001, train_matrix_nn,
      zero_train_matrix_nn, valid_data_nn, 41)  # returns list of accuracies over all epochs
# TODO: get predictions

# average the predictions

# evaluate the redictions
