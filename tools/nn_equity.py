import gym
from gym_env.env import HoldemTable
import numpy as np
# from tools.montecarlo_python import get_equity
from sklearn.preprocessing import OneHotEncoder
import time
from tools.nn_equity_model import equity_prediction_nn
import optuna
import pickle
import glob
import cppimport

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns



def make_one_hot_encoders():
    ranks = ['0', '2','3','4', '5', '6', '7', 
    '8', '9', 'T', 'J', 'Q', 'K','A']
    ranks = [[x] for x in ranks]
    suits = ['C', 'D', 'H', 'S', 'N']
    suits = [[x] for x in suits]

    rank_enc = OneHotEncoder(drop='first', sparse=False)
    rank_enc.fit(ranks)

    suits_enc = OneHotEncoder(drop='first', sparse=False)
    suits_enc.fit(suits)


    return rank_enc, suits_enc

def sample_cards(deck, num_cards):
    sampled_cards = []
    for _ in range(num_cards):
        card = np.random.randint(0, len(deck))
        sampled_cards.append(deck.pop(card))

    return sampled_cards

def preprocess_data_state(player_cards, table_cards, rank_enc, suit_enc):
    player_cards = list(player_cards)
    table_cards = list(table_cards)

    table_cards.extend(['0N']* (5 - len(table_cards)))
    table_card_ranks = [x[0] for x in table_cards]
    table_card_suits = [x[1] for x in table_cards]
    
    player_card_ranks = [x[0] for x in player_cards]
    player_card_suits = [x[1] for x in player_cards]
    
    encoded_table_card_ranks = [rank_enc.transform(np.array(x).reshape(1, -1)) for x in table_card_ranks]
    encoded_player_card_ranks = [rank_enc.transform(np.array(x).reshape(1, -1)) for x in player_card_ranks]

    encoded_table_card_ranks = np.array(encoded_table_card_ranks).ravel()
    encoded_player_card_ranks = np.array(encoded_player_card_ranks).ravel()

    state = np.concatenate((encoded_table_card_ranks, encoded_player_card_ranks))
    
    return state


def generate_data(n_samples=1000, max_runs=1000, write_data_dir=None, write_data_suffix=None):
    print("Generating data")
    cpp_calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
    cpp_equity_func = cpp_calculator.montecarlo
    get_equity = cpp_equity_func

    table = HoldemTable()
    rank_enc, suit_enc = make_one_hot_encoders()
    n = 0
    x_data = []
    y_data = []
    for i in range(n_samples):
        # Create deck
        table._create_card_deck()

        # Sample player cards
        p1_cards = sample_cards(table.deck, 2)
        p2_cards = sample_cards(table.deck, 2)

        # Sample table cards from either preflop,
        # flop, river or turn
        stage_card_nums = [0, 3, 4, 5]
        num_table_samples = np.random.choice(stage_card_nums)
        table_cards = sample_cards(table.deck, num_table_samples)
        
        equity = get_equity(set(p1_cards), set(table_cards), 2, max_runs)

        encoded_state = preprocess_data_state(p1_cards, table_cards, rank_enc, suit_enc)

        x_data.append(encoded_state)
        y_data.append(equity)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if write_data_dir and write_data_suffix:
        X_data_path = write_data_dir + 'X_' + write_data_suffix
        Y_data_path = write_data_dir + 'Y_' + write_data_suffix

        with open(X_data_path, 'wb') as handle:
            pickle.dump(x_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(Y_data_path, 'wb') as handle:
            pickle.dump(y_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return x_data, y_data


def load_train_data(train_data_dir, validation_train_split):
    print("loading training data")
    X_train_data_path_list = glob.glob(train_data_dir + 'X_train_data_*')
    Y_train_data_path_list = glob.glob(train_data_dir + 'Y_train_data_*')
    
    data_idx_list = []
    for x_data_path in X_train_data_path_list:
        idx = x_data_path.split('_')[-1]
        data_idx_list.append(idx)

    # data_idx_list.shuffle()

    X_data_list = []
    Y_data_list = []

    for idx in data_idx_list:
        X_data_path = train_data_dir + 'X_train_data_' + idx
        Y_data_path = train_data_dir + 'Y_train_data_' + idx

        with open(X_data_path, 'rb') as handle:
            X = pickle.load(handle)

        X_data_list.append(X)

        with open(Y_data_path, 'rb') as handle:
            Y = pickle.load(handle)

        Y_data_list.append(Y)

    X_data = np.concatenate(X_data_list)
    Y_data = np.concatenate(Y_data_list)

    n_data_sets = len(X_data)
    test_data_idx = int(validation_train_split * n_data_sets)

    X_test_data = X_data[0:test_data_idx]
    Y_test_data = Y_data[0:test_data_idx]

    X_train_data = X_data[test_data_idx:]
    Y_train_data = Y_data[test_data_idx:]

    return X_train_data, Y_train_data, X_test_data, Y_test_data


def optimisation_objective(trial):
    train_data_dir = "./nn_equity_model/train_data/"
    max_runs=1000
    training_episodes = 5
    evaluation_samples= 10000
    validation_train_split = 0.2


    X_train_data, Y_train_data, X_test_data, Y_test_data = load_train_data(train_data_dir, validation_train_split)

    # lr = trial.suggest_loguniform('lr', 1e-3, 1e-2)
    lr = 1e-3
    # batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 16, 32, 64, 128])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # batch_size = 60
    # num_epochs = trial.suggest_int('num_epochs', 1, 200)
    num_epochs = trial.suggest_int('num_epochs', 30, 80)
    layer_size = trial.suggest_categorical('layer_size', [128, 256, 512])

    print("New trial: ")
    print("batch_size: ", batch_size)
    print("num_epochs: ", num_epochs)
    print("layer_size: ", layer_size)
    print("")

    model_name = "equity_optuna_4_"+ str(trial._trial_id)
    # Initialise model
    model = equity_prediction_nn.Model(
        name=model_name, load_model=False, model_dir='./nn_equity_model/',
        observation_shape=X_train_data[0].shape, learning_rate=lr, layer_size=layer_size)

    model.train(X_train_data, Y_train_data, num_epochs=num_epochs, batch_size=batch_size)

    hist = model.evaluate(X_test_data, Y_test_data)
    mae = hist[1]

    return mae

def optuna_study(new_study_path=None, load_study=None):
    if load_study:
        pickle_out = load_study
        with open(load_study, 'rb') as handle:
            study = pickle.load(handle)
        print(study.trials_dataframe())
        print(" Value: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        exit()

    elif new_study_path:
        pickle_out = new_study_path

        study = optuna.create_study(direction="minimize")

    for i in range(1, 100):
        study.optimize(optimisation_objective, n_trials=1)

        with open(pickle_out, 'wb') as handle:
            pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(study.trials_dataframe())

def basic_train_routine(load_training_data=True):
    train_data_dir = "./nn_equity_model/train_data/"
    model_dir = "./nn_equity_model/"
    X_data, Y_data = generate_data(n_samples=1)
    model = equity_prediction_nn.Model(
        name="equity_2", load_model=True, model_dir=model_dir,
        observation_shape=X_data[0].shape)
    Y_pred = model.predict([X_data])

    batch_size = 64
    num_epochs = 500
    validation_train_split=0.0

    if load_training_data:
        X_train_data, Y_train_data, X_test_data, Y_test_data = load_train_data(train_data_dir, validation_train_split)
        model.train(X_train_data, Y_train_data, num_epochs=num_epochs, batch_size=batch_size)

    else:
        for i in range(88, 1000):

            train_data_suffix = "train_data" + "_" + str(i)
            print(train_data_suffix)

            X_data, Y_data = generate_data(n_samples=10000, max_runs=1000, 
                write_data_dir=train_data_dir, write_data_suffix=train_data_suffix)

            model.train(X_train_data, Y_train_data, num_epochs=num_epochs, batch_size=batch_size)


def plot_study_contour(study_path, output_dir):
    with open(study_path, 'rb') as handle:
        study = pickle.load(handle)

    print(study.trials_dataframe())

    fig = optuna.visualization.plot_contour(study, 
        params=['batch_size', 'layer_size'])

    # fig = fig.update_yaxes(categoryarray= [512, 256, 128]) 
    x = fig.to_ordered_dict()
    # print(x)
    fig.show()

    fig = optuna.visualization.plot_contour(study, 
        params=['num_epochs', 'layer_size'])

    # fig = fig.update_yaxes(categoryarray= [512, 256, 128]) 
    x = fig.to_ordered_dict()
    # print(x)
    fig.show()

    # fig = fig.update_yaxes(categoryarray= [512, 256, 128]) 
    # x = fig.to_ordered_dict()
    # # print(x)
    # fig.show()


class PredictEquity:
    def __init__(self, load_model_name=None, load_model_dir=None):
        
        X_data, Y_data = generate_data(n_samples=1)

        self.model = equity_prediction_nn.Model(name=load_model_name, load_model=True, model_dir=load_model_dir, observation_shape=X_data[0].shape)

        rank_enc, suit_enc = make_one_hot_encoders()

        self.rank_enc = rank_enc
        self.suit_enc = suit_enc


    def get_equity(self, player_cards, table_cards, players, runs=None):
        encoded_state = preprocess_data_state(player_cards, table_cards, self.rank_enc, self.suit_enc)
        encoded_state = encoded_state.reshape(1,-1)

        equity = self.model.predict(encoded_state)[0][0]

        return equity


if __name__ == "__main__":
    study_path = "./nn_equity_model/optuna_study_3.pickle"
    output_dir = "./nn_equity_model/equity_models/"
    # basic_train_routine(load_training_data=True)
    optuna_study(load_study=study_path)
    # plot_study_contour(study_path, output_dir)
    # optuna_study()
