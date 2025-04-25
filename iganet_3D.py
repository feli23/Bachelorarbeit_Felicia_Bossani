import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore
import pandas as pd
import optuna
import sys
import matplotlib.pyplot as plt
from time import time

DEVICE = "cpu"

if len(sys.argv) > 1:
    if sys.argv[1] == "cuda":
        DEVICE = "cuda"


# input = 3
# size_trainSet = 100
# size_testSet = 100

input = 3
ratio = 2
size_trainSet = 200 // ratio
size_testSet = 200 - size_trainSet

# input = 3
# size_trainSet = 22 // 3 + 1
# size_testSet = 22 - size_trainSet

dim_red = size_trainSet

# Read data
data_train = []
for i in range(100):
    # An deinen Pfad anpassen
    d = pd.read_csv('.\\Data_Motor_3d_train\\sol_Agap_fine_3d' + str(i+1) + '.csv', header=None) # new
    #d = pd.read_csv('.\\Data_Motor\\sol_Agap_fine_' + str(i) + '.csv', header=None)
    #d = pd.read_csv('.\\Data_Motor_3d\\sol_Agap_fine_3d' + str(i+1) + '.csv', header=None)
    x = np.reshape(d.values, 462)
    data_train.append(x)

data_test = []
for i in range(100):
#     # An deinen Pfad anpassen
    d = pd.read_csv('.\\Data_Motor_3d_test\\sol_Agap_fine_3d' + str(i+1) + '.csv', header=None) # new
#     #d = pd.read_csv('.\\Data_Motor\\sol_A gap_fine_' + str(i+1) + '.csv', header=None)
#     # d = pd.read_csv('C:\\work\\python\\firstNN\\pythonProject\\Data_Motor_3d\\sol_Agap_fine_3d' + str(i+1) + '.csv', header=None)
    x = np.reshape(d.values, 462)
    data_test.append(x)

#print(data_test)
input_train = pd.read_csv('.\\Data_Motor_3d_train\\input.csv', header=None).values
input_test = pd.read_csv('.\\Data_Motor_3d_test\\input.csv', header=None).values
# input_m = pd.read_csv('.\\Data_Motor_3d\\input.csv', header=None).values

# Read matrix Q
# An deinen Pfad anpassen
matrix = pd.read_csv('.\\MatrixA.csv', header=None)
# matrix = pd.read_csv('C:\\work\\python\\firstNN\\pythonProject\\MatrixA.csv', header=None)
matrix32 = matrix.astype('float32')
q = (1e10 * torch.tensor(matrix32.values, requires_grad=False, device=DEVICE))

print("running on: " + str(q.device))

for i in range(len(data_train)):
    data_test.append(data_train[i])

input_test = input_test.tolist()
for i in range(len(input_train)):
    input_test.append(input_train[i])


# Normalize data zscore
data_mean = np.mean(data_train)
data_std = np.std(data_train)
# data_norm = data - data_mean
data_norm = zscore(data_train, axis=None)

# Normalize date minmax
# data_min = np.min(data)
# data_max = np.max(data)
# data_norm = (data - data_min) / (data_max - data_min)


# Split Test Train Data


X_train = torch.Tensor(input_train).to(DEVICE)
X_test = torch.Tensor(input_test).to(DEVICE)
y_train = torch.Tensor(data_norm).to(DEVICE)
y_test = torch.Tensor(data_test).to(DEVICE)

# counter = 0
# count = 0
# # p = (2 * np.pi * 20) / 360
# X_train = torch.empty(size=[size_trainSet, input]).to(DEVICE)
# X_test = torch.empty(size=[22, input]).to(DEVICE)
# y_train = torch.empty(size=[size_trainSet, 462]).to(DEVICE)
# y_test_norm = torch.empty(size=[22, 462]).to(DEVICE)

# b = (2 * np.pi) / p
# b = 1
#
# # for i in range(200):
# #     alpha_rad = (2 * np.pi * i) / 360
# #     cos = np.cos(b * alpha_rad)
# #     sin = np.sin(b * alpha_rad)
# #     if i % ratio == 0:
# #         X_train[counter] = torch.tensor(input_m[i], device=DEVICE)
# #         y_train[counter] = torch.tensor(data_norm[i], device=DEVICE)
# #         counter = counter + 1
#
# #     X_test[i] = torch.tensor(input_m[i], device=DEVICE)
# #     y_test_norm[i] = torch.tensor(data_norm[i], device=DEVICE)
#
# p = (2 * np.pi * 20) / 360
# X_train = torch.empty(size=[size_trainSet, input])
# X_test = torch.empty(size=[200, input])
# y_train = torch.empty(size=[size_trainSet, 462])
# y_test_norm = torch.empty(size=[200, 462])
#
# for i in range(200):
#     alpha_rad = (2 * np.pi * i) / 360
#     cos = np.cos(b * alpha_rad)
#     sin = np.sin(b * alpha_rad)
#     if i % ratio == 0:
#         X_train[counter] = torch.tensor([cos, sin, i], device=DEVICE)
#         y_train[counter] = torch.tensor(data_norm[i], device=DEVICE)
#         counter = counter + 1
#     #else:
#     #        X_test[count] = torch.FloatTensor([cos, sin, i])
#     #        y_test[count] = torch.FloatTensor(data_norm[i])
#     #        count = count + 1
#     X_test[i] = torch.tensor([cos, sin, i], device=DEVICE)
#     y_test_norm[i] = torch.tensor(data_norm[i], device=DEVICE)

# # -- new impl --

# Proper Orthogonal Decomposition
# X = uSv^T
data32 = data_norm.astype('float32')
data_norm = torch.tensor(data32)
data_norm_t = torch.transpose(data_norm, 0, 1)
y_train_t = torch.transpose(y_train, 0, 1).cpu()
[u, s, vt] = np.linalg.svd(y_train_t, full_matrices=False, compute_uv=True)  # u(462, 22), v(22, 22), s(22, 1)
y_train_t = y_train_t.to(DEVICE)

# Transform y_train
u_red = torch.FloatTensor(u[:, :dim_red]).to(DEVICE)  # u.clone().detach()
y_train_red_t = torch.matmul(torch.transpose(u_red, 0, 1), y_train_t)
y_train_red = torch.transpose(y_train_red_t, 0, 1)


def lossf(rho_i, res_i):
    # print(": " + str(rho_i)  + " - " + str(res_i))
    temp = rho_i - res_i
    loss_train_i = torch.div(torch.matmul(torch.matmul(temp, q), temp), torch.matmul(torch.matmul(rho_i, q), rho_i))
    return loss_train_i

def lossfunction_l2(res, rho, q):  # res und rho sollten liegen, bzw (#sampels x #dofs)
    # losses_train = torch.empty(len(res), device=DEVICE, dtype=torch.float32)
    # with Pool(20) as pool:
    #     lt = pool.starmap(lossf, zip(rho.cpu().detach(), res.cpu().detach()))
        # pool.close()
        # pool.join()
    lt = map(lossf, rho, res)
    # print(fmean(torch.stack(list(lt))))
    # print("map: " + str(lt))
    return torch.mean(torch.stack(list(lt))) # mean


def lossfunction_l2_test_train(res, rho, q):
    # s1 = time()
    loss_train = lossfunction_l2(res, rho, q)
    # print(time() - s1)
    return loss_train

def compute_l2(res, rho):
    temp = rho - res
    mul1 = torch.matmul(temp, q)
    dividend = torch.matmul(mul1, temp)
    mul2 = torch.matmul(rho, q)
    divisor = torch.matmul(mul2, rho)
    loss = torch.div(dividend, divisor)
    loss_rel = torch.sqrt(loss)
    return loss_rel

def calc_test_loss(model):
    i = 0
    for i in range(0,10):
        start = time()
        output = model(X_test)
        print("t" + str(i) + ": " +str(time()-start))

    y_e_red = torch.transpose(output, 0, 1)
    y_e = torch.transpose(torch.matmul(u_red, y_e_red), 0, 1)

    y_eval = (y_e * data_std) + data_mean
    #y_test = (y_test_norm * data_std) + data_mean

    lt = map(compute_l2, y_eval, y_test)
    out = torch.stack(list(lt))
    #print(torch.mean(out))
    return out # should be < 1% => <0.01

#    return compute_l2(y_eval.t(), y_test, q)

# -- mse below --
# def lossfunction_l2_test_train(res, rho, q): #old
#     loss_train = F.mse_loss(res, rho, reduction='mean')
#     loss_train_rel = loss_train / F.mse_loss(rho, torch.zeros_like(rho), reduction='mean')
#     return loss_train_rel
#
# def compute_l2(res, rho):
#     err = F.mse_loss(res, rho, reduction='mean')
#     l2 = np.sqrt(err.cpu().detach().numpy())
#     len_rho = F.mse_loss(rho, torch.zeros_like(rho), reduction='mean')
#     l2_rel = l2 / np.sqrt(len_rho.cpu())
#     return l2_rel
#
# def calc_test_loss(model):
#     y_e_norm = model(X_test)
#     y_e = (y_e_norm * data_std) + data_mean
#     y_test = (y_test_norm * data_std) + data_mean
#
#     # loss_test = F.mse_loss(y_e, y_test, reduction='none')
#     # print(loss_test)
#     # out = loss_test / F.mse_loss(y_test, torch.zeros_like(y_test), reduction='none')
#     # out = np.sqrt(out.cpu().detach().numpy())
#
#     out = torch.empty(22)
#     for i in range(22):
#             #err_v = compute_mse(y_eval[i], y_test[i])
#             err_v = compute_l2(y_e[i], y_test[i])
#             out[i] = err_v
#
#     return out
# -- mse above --

#     # y_e = torch.transpose(model(X_test), 0, 1)
#     # # y_e = torch.matmul(u_red, y_e_red)
#
#     # y_eval = (y_e * data_std) + data_mean
#     # y_test = (y_test_norm * data_std) + data_mean
#
#     # lt = map(compute_l2, y_eval.t(), y_test)
#     # out = torch.stack(list(lt))
#     # print(torch.mean(out))
#     return out # should be < 1% => <0.01
#
#     # return compute_l2(y_eval.t(), y_test, q)

# Train model
def train(optimizer, epochs, model, trial = None):
    # losses_train = []
    # losses_test = []
    # q_red = u_red.t() @ q @ u_red
    # y_test_norm_red = u_red.t() @ y_test_norm.t();
    for i in range(epochs):
        #y_pred = model.forward(X_train)
        # X_train.device
        y_pred_red = torch.transpose(model(X_train), 0, 1)  # get prediction
        y_pred = torch.matmul(u_red, y_pred_red)  # Reconstruct full dimension
        y_pred = y_pred.to(DEVICE)
        # print(y_pred.device)
        # loss_train, loss_test = lossfunction_mse_test_train(y_pred_red, y_train_red, X_test, y_test_norm, model, u_red)
        # s1 = time.time()
        #  loss_train, loss_test = lossfunction_l2_test_train(y_pred_red, y_train_red_t, X_test, y_test_norm_red, model, q_red, u_red)
        loss_train = lossfunction_l2_test_train(y_pred.t(), y_train, q) #für pod .t() hinzufügen bei y_pred
        # print("s1: " + str(time.time() - s1))
        # s1 = time.time()

        # losses_train.append(loss_train)
        # losses_test.append(loss_test)
        # print("s2: " + str(time.time() - s1))
        if (i + 1) % 200 == 0:
            loss_test = calc_test_loss(model)
            if (trial):
                trial.report(torch.mean(loss_test), i)
            else:
                print(str(torch.mean(loss_test)) + str(i))
            # trial.report(loss_test, i)
            print(f'Train Epoch: {i} Trainingsloss: {loss_train}')
            # print(f'Train Epoch: {i} Validationloss: {loss_test}')

        # s1 = time.time()
        # Do some back propagation
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        # print("s3: " + str(time.time() - s1))
    
    output = calc_test_loss(model)
    loss_test = torch.mean(output)
    biggest = torch.topk(output, 5)

    # loss_test = calc_test_loss(model)

    if (trial):
        print(torch.mean(output).item())
        trial.set_user_attr("max_test_loss", torch.max(output).item())
        trial.set_user_attr("var_test_loss", torch.var(output).item())
        trial.set_user_attr("max_test_losses", biggest.values.cpu().detach().numpy().tolist())
        trial.set_user_attr("loss_train", loss_train.item())
    else:
        print("max_test_loss "+ str(torch.max(output).item()))
        print("var_test_loss "+ str(torch.var(output).item()))
        print("max_test_losses "+ str(biggest.values.cpu().detach().numpy().tolist()))
        print("loss_train "+ str(loss_train.item()))
        # visual_rel_err(output)
    # print(loss_test)
    # print(loss_test.item())

    print('mean: ' + str(np.mean(output.detach().numpy())))
    print('std: ' + str(np.std(output.detach().numpy())))
    print('max: ' + str(np.max(output.detach().numpy())))

    return loss_test.item() #losses_train[len(losses_train)-1].item(), losses_test[len(losses_train)-1].item()

# epochs = 5000
# lr = 1e-4
# wd = 1e-7
# seed = 100
#
# torch.manual_seed(seed)
# model = ModelDNN()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# losses_train, losses_test = train()
# err_val, y_eval = evaluate()
# err_val_mean = np.mean(err_val)
# err_val_std = np.std(err_val)
# err_val_max = np.max(err_val)
#
# temp = open('C:\\Users\\felic\\git\\Bachelorarbeit\\ParameterTest_POD_3D_zscore\\Error_Validation_Epochen_' + str(epochs) + '_LR_' + str(lr) + '_WD_' + str(wd) + '.csv', 'a')
# #temp = open('C:\\work\\python\\firstNN\\pythonProject\\ParameterTest_POD_3D_zscore\\Error_Validation_Epochen_' + str(epochs) + '_LR_' + str(lr) + '_WD_' + str(wd) + '.csv', 'a')
# #
# for k in range(len(err_val)):
#     temp.write('Sample' + str(k) + ': ' + str(err_val[k]) + '\n')
# temp.write('Durchschnitt: ' + str(err_val_mean) + '\n')
# temp.write('Varianz: ' + str(err_val_std) + '\n')
# temp.write('Maximum: ' + str(err_val_max))
#
# visual_loss(losses_train, losses_test, epochs, lr, wd)
# visual_rel_err(err_val)

# Konstanten
wds = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
epoch_boundries = [1000, 10000, 250]
layer_sizes = [[5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024], [5, 1024]]
layers_boundries = [3, 10]
MANUAL = True

def run():
    learning_rate = 1e-4
    epochs = 6250
    wdecay = 1e-5

    torch.manual_seed(100)
    # model = ModelDNN()
    model = define_model_m(6, [9, 360, 440, 105, 510, 495]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wdecay)
    
    loss_test = train(optimizer, epochs, model)
    output = calc_test_loss(model)

    # for j in range(len(output)):
    #     temp = open('.\\PredictedSol\\pred_sol_' + str(j) + '.csv', 'a')
    #     temp.write(str(output[j]))

    #visual_rel_err(output.detach().numpy())

    #print(loss_test)


def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", lrs)
    epochs = trial.suggest_int("epochs", epoch_boundries[0], epoch_boundries[1], step=epoch_boundries[2])
    wdecay = trial.suggest_categorical("weight_decay", wds)

    torch.manual_seed(100)
    # model = ModelDNN()
    model = define_model(trial).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wdecay)
    
    loss_test = train(optimizer, epochs, model, trial)
    return loss_test

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]

def visual_rel_err(err_val):
    plt.plot(range(22), err_val, color='r', label='test')
    plt.ylabel('relative error')
    plt.xlabel('Rotation Angle')
    plt.legend()
    # plt.savefig('.\\RelErr_test8_bestTrail_1.pdf', bbox_inches='tight')
    plt.show()

def define_model_m(n_layers, out_features):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    layers = []
    in_features = 3
    for i in range(n_layers):
        # layertype = trial.suggest_categorical("layer_type_l{}".format(i), ["linear", "lstm"])
        # if layertype == "linear":
        layers.append(nn.Linear(in_features, out_features[i]))
        layers.append(nn.ReLU())
        # else:
            # kernel = trial.suggest_int("n_units_l{}_groups".format(i), 2, 4)
            # layers.append(nn.LSTM(in_features, out_features,1, batch_first = True))
            # layers.append(extract_tensor())

        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        # layers.append(nn.Dropout(p))

        in_features = out_features[i]
    layers.append(nn.Linear(in_features, dim_red)) #with POD: 100; without POD: 462
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", layers_boundries[0], layers_boundries[1])
    layers = []
    

    in_features = 3
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), layer_sizes[i][0], layer_sizes[i][1], step=5)
        # layertype = trial.suggest_categorical("layer_type_l{}".format(i), ["linear", "lstm"])
        # if layertype == "linear":
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # else:
            # kernel = trial.suggest_int("n_units_l{}_groups".format(i), 2, 4)
            # layers.append(nn.LSTM(in_features, out_features,1, batch_first = True))
            # layers.append(extract_tensor())

        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        # layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, 462)) #with POD: 100; without POD: 462
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


if __name__ == "__main__":
    if (MANUAL):
        run()
    else:
        # createStudy()
        storage_name = "sqlite:///example.db"
        study = optuna.create_study(study_name="test10_winkel_new", direction="minimize", sampler=optuna.samplers.TPESampler(), storage=storage_name, load_if_exists=True)

        study.set_user_attr("learning_rates", lrs)
        study.set_user_attr("weight_decays", wds)
        study.set_user_attr("epochs", epoch_boundries)
        study.set_user_attr("layer_sizes", layer_sizes)
        study.set_user_attr("n_layers", layers_boundries)
        study.set_user_attr("comment", "without POD calculate loss based on mean with diff. layertypes")

        study.optimize(objective, n_trials=1000)

