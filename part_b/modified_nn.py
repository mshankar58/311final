import matplotlib.pyplot as plt
import scipy.sparse

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

train_data = load_train_csv("../data")


# modified helper function from utils.py
def _load_student_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "gender": [],
        "premium_pupil": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if reader.line_num != 1:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
                if row[3] == '':
                    data["premium_pupil"].append(-1)
                else:
                    data["premium_pupil"].append(int(float(row[3])))

    return data


# modified load function from example in utils.py
def load_student_meta_csv(root_dir="/data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "student_meta.csv")
    return _load_student_csv(path)


# implemented function: split the dataset by gender
def split_by_gender(train_matrix, base_path="../data"):
    # three types of gender (0, 1, 2)
    # will create three gender dictionaries
    g_0 = {
        "user_id": [],
    }
    g_1 = {
        "user_id": [],
    }
    g_2 = {
        "user_id": [],
    }

    metadata = load_student_meta_csv(base_path)
    num_stu = len(metadata["user_id"])
    for s in range(num_stu):
        user_id = metadata["user_id"][s]
        gender = metadata["gender"][s]
        # type 0
        if gender == 0:
            g_0["user_id"].append(user_id)
        # type 1
        if gender == 1:
            g_1["user_id"].append(user_id)
        # type 2
        if gender == 2:
            g_2["user_id"].append(user_id)
    total_num = train_matrix.shape[0]
    matrix_0 = None
    matrix_1 = None
    matrix_2 = None
    for i in range(total_num):
        student = metadata["user_id"][i]
        if student in g_0["user_id"]:
            if matrix_0 is None:
                matrix_0 = scipy.sparse.csr_matrix(train_matrix[i])
            else:
                scipy.sparse.vstack([matrix_0, scipy.sparse.csr_matrix(train_matrix[i])])
        if student in g_1["user_id"]:
            if matrix_1 is None:
                matrix_1 = scipy.sparse.csr_matrix(train_matrix[i])
            else:
                scipy.sparse.vstack([matrix_1, scipy.sparse.csr_matrix(train_matrix[i])])
        if student in g_2["user_id"]:
            if matrix_2 is None:
                matrix_2 = scipy.sparse.csr_matrix(train_matrix[i])
            else:
                scipy.sparse.vstack([matrix_2, scipy.sparse.csr_matrix(train_matrix[i])])
    return g_0, g_1, g_2


# split by premium pupil
def split_by_premium(train_matrix, base_path="../data"):
    # "1.0" true as a premium pupil, "0.0" false
    # split the data into two dictionaries
    p_0 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    p_1 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    p_2 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    matrix_0 = None
    matrix_1 = None
    matrix_2 = None
    metadata = load_student_meta_csv(base_path)
    num_stu = len(metadata["user_id"])
    for s in range(num_stu):
        user_id = metadata["user_id"][s]
        premium_pupil = metadata["premium_pupil"][s]
        # premium pupil data not available
        if premium_pupil == - 1:
            p_2["user_id"].append(user_id)
        # student is not a premium pupil
        elif premium_pupil == 0:
            p_0["user_id"].append(user_id)
        # student is a premium pupil
        else:
            p_1["user_id"].append(user_id)

    for student in p_0["user_id"]:
        if matrix_0 is None:
            matrix_0 = train_matrix[student]
        else:
            torch.vstack((matrix_0, train_matrix[student]))
    for student in p_1["user_id"]:
        if matrix_1 is None:
            matrix_1 = train_matrix[student]
        else:
            torch.vstack((matrix_1, train_matrix[student]))
    for student in p_2["user_id"]:
        if matrix_2 is None:
            matrix_2 = train_matrix[student]
        else:
            torch.vstack((matrix_2, train_matrix[student]))

    return matrix_0, matrix_1, matrix_2


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.hidden1 = nn.Linear(num_question, k)
        self.hidden2 = nn.Linear(k, k)
        self.hidden3 = nn.Linear(k, k)
        self.output = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        h1_w_norm = torch.norm(self.hidden1.weight, 2) ** 2
        h2_w_norm = torch.norm(self.hidden2.weight, 2) ** 2
        h3_w_norm = torch.norm(self.hidden2.weight, 2) ** 2
        out_w_norm = torch.norm(self.output.weight, 2) ** 2
        return h1_w_norm + h2_w_norm + out_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        hidden1 = self.hidden1(inputs)
        m = nn.ReLU()
        hidden1 = m(hidden1)

        hidden2 = self.hidden2(hidden1)
        n = nn.ReLU()
        hidden2 = n(hidden2)

        # hidden3 = self.hidden2(hidden2)
        # p = nn.ReLU()
        # hidden3 = p(hidden3)


        # h_drop = F.dropout(hidden3, p=0.5, training=True)
        out = self.output(hidden2)
        q = nn.Sigmoid()
        out = q(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_matrix, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]
    valid = []
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            r = 0.5 * model.get_weight_norm()

            loss = torch.sum((output - target) ** 2.) + (lamb * r)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t ""Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return valid
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    # p_0 represents non premium pupil, p_1 premium pupil, p_2 premium pupil data not available
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    # m_0, m_1, m_2 = split_by_premium(train_matrix)
    # g_0, g_1, g_2 = split_by_gender(train_matrix=train_matrix)

    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 50
    model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 20
    lamb = 0.001
    # validation accuracy for g_0
    # t_0 = train(model, lr, lamb, g_0, zero_train_matrix, valid_data, num_epoch)
    # validation accuracy for g_1
    # t_1 = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # validation accuracy for g_2
    # t_2 = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    # trial with split by premium pupil
    t_0 = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # t_1 = train(model, lr, lamb, m_1, zero_train_matrix, valid_data, num_epoch)
    # t_2 = train(model, lr, lamb, m_2, zero_train_matrix, valid_data, num_epoch)

    plt.figure()
    plt.plot(t_0)
    # plt.plot(t_1)
    # plt.plot(t_2)
    plt.title("Accuracy over iterations on validation set")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
