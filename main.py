# Author: Siqian Hou
# Date: 03/02/2023
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob

FEATURE_SIZE = 8
TRAIN_SIZE = 900
TEST_SIZE = 130
EPOCH_SIZE = 100  # basic epoch size for learning rate tuning
EPOCH_INCREASE_SIZE = 2  # increase the basic epoch size by this size in the actual training process
CONVERGE_EPOCH_SIZE = 10  # converge epoch size to check whether the mse loss is converging
CONVERGE_THRESHOLD = 10 ** -9  # mse loss change threshold between each epoch to check convergence
START_LEARNING_RATE = 100
LEARNING_RATE_RANGE_1 = 10  # the number of times we try the learning rate (divided by 10): 100, 10, 0.1...
LEARNING_RATE_RANGE_2 = 30  # the number of times we try the learning rate (add prev_best_lr/10): 0.1, 0.11, 0.12...


# Get train test split
def my_train_test_split(all_data_df):
    for _i in range(FEATURE_SIZE, FEATURE_SIZE * 2):
        all_data_df.insert(_i,
                           "STD_" + standardization_data_df.columns[_i - FEATURE_SIZE],
                           standardization_data_df.iloc[:, _i - FEATURE_SIZE],
                           allow_duplicates=True)

    all_data_df.insert(FEATURE_SIZE * 2 + 1,
                       "STD_" + standardization_data_df.columns[FEATURE_SIZE],
                       standardization_data_df.iloc[:, FEATURE_SIZE],
                       allow_duplicates=True)

    # Separate training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        all_data_df.iloc[:, :FEATURE_SIZE * 2], all_data_df.iloc[:, FEATURE_SIZE * 2:], test_size=TEST_SIZE)

    # Unzip raw and standardized data
    _X_raw_train = X_train.drop(X_train.iloc[:, FEATURE_SIZE:FEATURE_SIZE * 2], axis=1)
    _X_std_train = X_train.drop(X_train.iloc[:, :FEATURE_SIZE], axis=1)
    _X_raw_test = X_test.drop(X_test.iloc[:, FEATURE_SIZE:FEATURE_SIZE * 2], axis=1)
    _X_std_test = X_test.drop(X_test.iloc[:, :FEATURE_SIZE], axis=1)
    _y_raw_train = y_train.drop(y_train.columns[1], axis=1)
    _y_std_train = y_train.drop(y_train.columns[0], axis=1)
    _y_raw_test = y_test.drop(y_test.columns[1], axis=1)
    _y_std_test = y_test.drop(y_test.columns[0], axis=1)

    return _X_raw_train, _X_std_train, _X_raw_test, _X_std_test, _y_raw_train, _y_std_train, _y_raw_test, _y_std_test


# Train the model using given learning rate, return the possible minimum loss of current model
# Default value for feature index is -1: for multivariate usage; otherwise, pass a particular feature index
# Default value for show plot is False: in the final program, only show plot in the final training process
# Default value for learning rate tuning is True: control epoch size and convergence checking process
def train(X_train, y_train, _learning_rate, feature_idx=-1, show_plt=False, lr_tuning=True):
    m_list = []
    if feature_idx == -1:
        m_list = [0] * FEATURE_SIZE
    _m, _b, converge_times = 0, 0, 0
    mse_loss_array = []

    # Increase epoch times during real training, low epoch for learning rate tuning only
    epoch_times = EPOCH_SIZE
    if not lr_tuning:
        epoch_times *= EPOCH_INCREASE_SIZE

    # Gradient descent iteration
    for _ in range(epoch_times):
        derivative_m_list = []
        if feature_idx == -1:
            derivative_m_list = [0] * FEATURE_SIZE
        derivative_m, derivative_b, _mse_loss = 0, 0, 0
        # Calculate updated m, b according to gradient descent formula
        for j in range(TRAIN_SIZE):
            _y_observe = y_train.iat[j, 0]
            _y_predict = 0
            if feature_idx == -1:
                for idx in range(FEATURE_SIZE):
                    _y_predict += m_list[idx] * X_train.iat[j, idx]
                _y_predict += _b
                for idx in range(FEATURE_SIZE):
                    derivative_m_list[idx] += -2 * X_train.iat[j, idx] * (_y_observe - _y_predict)
            else:
                _y_predict = _m * X_train.iat[j, feature_idx] + _b
                derivative_m += -2 * X_train.iat[j, feature_idx] * (_y_observe - _y_predict)
            derivative_b += -2 * (_y_observe - _y_predict)
            _mse_loss += (_y_observe - _y_predict) ** 2
        if feature_idx == -1:
            derivative_m_list = [dm / TRAIN_SIZE for dm in derivative_m_list]
            m_list = [m - _learning_rate * dm for m, dm in zip(m_list, derivative_m_list)]
        else:
            derivative_m /= TRAIN_SIZE
            _m -= _learning_rate * derivative_m
        derivative_b /= TRAIN_SIZE
        _b -= _learning_rate * derivative_b
        _mse_loss /= TRAIN_SIZE
        mse_loss_array.append(_mse_loss)

        if lr_tuning:
            # Only check if the mse is converging when doing learning rate tuning
            # If converged: break and record current loss
            # If not: continue until reach the epoch size and record the final loss
            if len(mse_loss_array) >= 2 and abs(mse_loss_array[-1] - mse_loss_array[-2]) <= CONVERGE_THRESHOLD:
                converge_times += 1
                if converge_times == CONVERGE_EPOCH_SIZE:
                    break
    print("lr: ", _learning_rate)
    print("mse: ", mse_loss_array)
    if show_plt:
        plt.plot(mse_loss_array)
        if feature_idx == -1:
            if y_train.columns[0][:3] == "STD":
                plt.title("STD lr: " + str(_learning_rate))
            else:
                plt.title("RAW lr: " + str(_learning_rate))
        else:
            plt.title(X_train.columns[feature_idx] + "\n lr: " + str(_learning_rate))
        plt.show()

    if feature_idx == -1:
        return [mse_loss_array[-1], m_list, _b]
    else:
        return [mse_loss_array[-1], _m, _b]


# R-Squared on training data
def r_squared_training(y_train, mse_train_loss):
    y_train_var = y_train.iloc[:, 0].var()
    rs_train = 1 - (mse_train_loss / y_train_var)
    print("Training R-Squared: ", rs_train)
    return rs_train


# R-Squared on testing data
# Default value for feature index is -1: for multivariate usage; otherwise, pass a particular feature index
def r_squared_testing(X_test, y_test, m, b, f=-1):
    y_test_var = y_test.iloc[:, 0].var()
    mse_test_loss = 0
    for i in range(TEST_SIZE):
        y_observe = y_test.iat[i, 0]
        y_predict = 0
        if f == -1:
            for idx in range(FEATURE_SIZE):
                y_predict += m[idx] * X_test.iat[i, idx]
            y_predict += b
        else:
            x = X_test.iat[i, f]
            y_predict = m * x + b
        mse_test_loss += (y_observe - y_predict) ** 2
    mse_test_loss /= TEST_SIZE
    rs_test = 1 - (mse_test_loss / y_test_var)
    print("Testing R-Squared: ", rs_test)
    return rs_test


# Wrap learning rate tuning, actual training process, r-squared calculation and drawing plots.
def train_helper(X_train, X_test, y_train, y_test, is_multivariate=False):
    results = []
    for f in range(FEATURE_SIZE):
        if not is_multivariate:
            print("---------- FEATURE: " + X_train.columns[f] + " ----------")
        # Hyperparameter tuning
        # Test different learning rate exponentially first
        learning_rate = START_LEARNING_RATE
        converge_loss_array = []
        for _ in range(LEARNING_RATE_RANGE_1):
            learning_rate *= 10 ** -1
            if is_multivariate:
                converge_loss_array.append(train(X_train, y_train, learning_rate)[0])
            else:
                converge_loss_array.append(train(X_train, y_train, learning_rate, feature_idx=f)[0])

        # Find previous best learning rate
        prev_min_loss = math.inf
        for loss in converge_loss_array:
            if not math.isnan(loss) and not math.isinf(loss):
                prev_min_loss = min(loss, prev_min_loss)
        prev_best_learning_rate_idx = converge_loss_array.index(prev_min_loss)
        prev_best_learning_rate = START_LEARNING_RATE * (10 ** -(prev_best_learning_rate_idx + 1))
        print("First round loss: ", converge_loss_array)
        print("First round best lr: ", prev_best_learning_rate)
        print("First round best loss: ", prev_min_loss)

        # Test different learning rate around previous best learning rate
        start_lr = prev_best_learning_rate / 2
        lr_increase = prev_best_learning_rate / 10
        converge_loss_array = []
        for i in range(LEARNING_RATE_RANGE_2):
            learning_rate = start_lr + i * lr_increase
            if is_multivariate:
                converge_loss_array.append(train(X_train, y_train, learning_rate)[0])
            else:
                converge_loss_array.append(train(X_train, y_train, learning_rate, feature_idx=f)[0])
        best_learning_rate = start_lr + (lr_increase * converge_loss_array.index(min(converge_loss_array)))
        print("Second round loss: ", converge_loss_array)
        print("Second round best lr: ", best_learning_rate)
        print("Second round best loss: ", min(converge_loss_array))

        # Train the final model using the best learning rate, stop until the epoch size is reached
        if is_multivariate:
            [mse_train_loss, m, b] = train(X_train, y_train, best_learning_rate,
                                           show_plt=True, lr_tuning=False)
        else:
            [mse_train_loss, m, b] = train(X_train, y_train, best_learning_rate, feature_idx=f,
                                           show_plt=True, lr_tuning=False)

        # R-Squared
        rs_train = r_squared_training(y_train, mse_train_loss)
        if is_multivariate:
            rs_test = r_squared_testing(X_test, y_test, m, b)
        else:
            rs_test = r_squared_testing(X_test, y_test, m, b, f=f)

        # Get data type
        prefix = y_train.columns[0][:3]
        # Scatter plot on training data with final model
        if not is_multivariate:
            plt.scatter(X_train.iloc[:, f], y_train)
            y_predict_train = X_train.iloc[:, f].copy()
            y_predict_train = y_predict_train.map(lambda n: n * m + b)
            plt.plot(X_train.iloc[:, f], y_predict_train, 'r')
            plt.title(X_train.columns[f] + "\n Training Data R-Squared: " + str(rs_train))

            feature_name = X_train.columns[f].split("(")[0]
            feature_name = feature_name.strip()
            feature_name = feature_name.replace(" ", "_")
            if prefix == "STD":
                plot_name = "figures/std/" + feature_name + ".png"
            else:
                plot_name = "figures/raw/" + feature_name + ".png"
            plt.savefig(plot_name)

            plt.show()

        if prefix == "STD":
            data_type = "All STD Features"
        else:
            data_type = "All RAW Features"

        model = {
            "feature": data_type if is_multivariate else X_train.columns[f],
            "learning rate": learning_rate,
            "m": m,
            "b": b,
            "rs_train": rs_train,
            "rs_test": rs_test
        }
        results.append(model)
        if is_multivariate:
            break
    return results


def result_to_string_helper(results):
    results_str = "[\n"
    for result in results:
        results_str += "{\n"
        for k, v in result.items():
            results_str += k
            results_str += ": "
            results_str += str(v)
            results_str += "\n"
        results_str += "}\n"
    results_str += "]\n"
    return results_str


# Main logic for the algorithm.
if __name__ == '__main__':
    # Read raw and preprocessed(standardized) date
    raw_data_df = pd.read_excel("data/Concrete_Data.xls")
    standardization_data_df = pd.read_excel("data/Standardization_Data.xls")

    # Get all required splits
    X_raw_train, X_std_train, X_raw_test, X_std_test, y_raw_train, y_std_train, y_raw_test, y_std_test = \
        my_train_test_split(raw_data_df.copy())

    # Set up program output for report usage
    plots = glob.glob("figures/raw/*")
    for p in plots:
        os.remove(p)
    plots = glob.glob("figures/std/*")
    for p in plots:
        os.remove(p)
    open("results.txt", "w").close()
    results_file = open("results.txt", "a")

    # Start algorithm for training different model
    print("**********---------- UNIVARIATE RAW ----------**********")
    univariate_raw_results = train_helper(X_raw_train, X_raw_test, y_raw_train, y_raw_test)
    results_file.write("**********---------- UNIVARIATE RAW ----------**********\n")
    results_file.write(result_to_string_helper(univariate_raw_results))

    print("**********---------- MULTIVARIATE RAW ----------**********")
    multivariate_raw_result = train_helper(X_raw_train, X_raw_test, y_raw_train, y_raw_test, is_multivariate=True)
    results_file.write("**********---------- MULTIVARIATE RAW ----------**********\n")
    results_file.write(result_to_string_helper(multivariate_raw_result))

    print("**********---------- UNIVARIATE STANDARDIZED ----------**********")
    univariate_std_results = train_helper(X_std_train, X_std_test, y_std_train, y_std_test)
    results_file.write("**********---------- UNIVARIATE STANDARDIZED ----------**********")
    results_file.write(result_to_string_helper(univariate_std_results))

    print("**********---------- MULTIVARIATE STANDARDIZED ----------**********")
    multivariate_std_result = train_helper(X_std_train, X_std_test, y_std_train, y_std_test, is_multivariate=True)
    results_file.write("**********---------- MULTIVARIATE STANDARDIZED ----------**********")
    results_file.write(result_to_string_helper(multivariate_std_result))

    print(univariate_raw_results)
    print(multivariate_raw_result)
    print(univariate_std_results)
    print(multivariate_std_result)
