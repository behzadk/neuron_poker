import time
from tools.montecarlo_python import get_equity as py_get_equity
import tools.nn_equity as nn_equity
from gym_env.env import HoldemTable
from tools.nn_equity import sample_cards
import numpy as np
import matplotlib.pyplot as plt
import cppimport


def test_model(get_equity_func, my_cards, cards_on_table, players, runs):
    start_time = time.time()
    res = get_equity_func(my_cards, cards_on_table, players, runs)
    end_time = time.time()

    execution_time = end_time - start_time

    return res, execution_time

def sample_scenario():
    table = HoldemTable()

    # Create deck
    table._create_card_deck()

    # Sample player cards
    p1_cards = sample_cards(table.deck, 2)
    p2_cards = sample_cards(table.deck, 2)

    # Sample table cards from either preflop,
    # flop, river or turn
    stage_card_nums = [0, 3, 4, 5]
    num_table_samples = np.random.choice(stage_card_nums)
    cards_on_table = sample_cards(table.deck, num_table_samples)

    my_cards = set(p1_cards)
    cards_on_table = set(cards_on_table)

    return my_cards, cards_on_table


def speed_and_error_comparison(number_of_samples):
    # python_equity_calculator = montecarlo_python.MonteCarlo()
    # py_equity_func = python_equity_calculator.run_montecarlo

    cpp_calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
    cpp_equity_func = cpp_calculator.montecarlo

    players = 2

    load_model = "equity_optuna_4_17"
    nn_equity_calculator = nn_equity.PredictEquity(load_model_name=load_model, load_model_dir='./tools/nn_equity_model/')
    nn_equity_func = nn_equity_calculator.get_equity

    model_data_dict = {'cpp_montecarlo_10k': {'equity_function': cpp_equity_func, 'runs': 10000, 'error_data': [], 'time_data': []},
                    'cpp_montecarlo_1k': {'equity_function': cpp_equity_func, 'runs': 1000, 'error_data': [], 'time_data': []},
                    # 'py_montecarlo_1k': {'equity_function': py_get_equity, 'runs': 1000, 'error_data': [], 'time_data': []},
                    'neural_network': {'equity_function': nn_equity_func, 'runs': 1, 'error_data': [], 'time_data': []}}

    for i in range(number_of_samples):
        print("Sample number {}".format(i))

        my_cards, cards_on_table = sample_scenario()
        base_line_res, base_line_time = test_model(cpp_equity_func, my_cards, cards_on_table, players, runs=100000)
        

        for key in model_data_dict.keys():
            equity_func = model_data_dict[key]['equity_function']
            runs = model_data_dict[key]['runs']
            res, ex_time = test_model(equity_func, my_cards, cards_on_table, players, runs=runs)
            error = abs(res - base_line_res)

            if error > 0.1 and key=='neural_network':
                print(my_cards)
                print(cards_on_table)

            # Don't save first results. Tensorflow is slow on first prediction use 
            # of a model
            if i == 0:
                continue

            print("Name: {}, \tError: {}, \tTime: {}".format(key, error, ex_time))


            model_data_dict[key]['error_data'].append(error)
            model_data_dict[key]['time_data'].append(ex_time)
    

    print(model_data_dict['neural_network']['time_data'])
    return model_data_dict


def plot_model_comparison(model_data_dict):
    model_names = model_data_dict.keys()
    # Calculate statistics
    for key in model_names:
        model_errors = model_data_dict[key]['error_data']
        model_times = model_data_dict[key]['time_data']

        model_data_dict[key]['mean_error'] = np.mean(model_errors)
        model_data_dict[key]['std_error'] = np.std(model_errors)

        model_data_dict[key]['mean_time'] = np.mean(model_times)
        model_data_dict[key]['std_time'] = np.std(model_times)

    model_mean_errors = [model_data_dict[key]['mean_error'] for key in model_names]
    model_error_stdevs = [model_data_dict[key]['std_error'] for key in model_names]

    ind = np.arange(len(model_mean_errors))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, model_mean_errors, width, yerr=model_error_stdevs,
                    color='SkyBlue', label='time')
    ax.set_ylabel('Absolute equity approximation error (compared to montecarlo 100k runs)')
    ax.set_xlabel('Model')
    ax.set_xticks(ind)
    ax.set_xticklabels(model_names)
    plt.savefig('./error_comparison.png', dpi=100)
    plt.show()
    plt.close()


    model_mean_time = [model_data_dict[key]['mean_time'] for key in model_names]
    model_time_stdevs = [model_data_dict[key]['std_time'] for key in model_names]

    fig, ax = plt.subplots()
    rects2 = ax.bar(ind, model_mean_time, width, yerr=model_time_stdevs,
                    color='IndianRed', label='error')
    ax.set_xticks(ind)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Average computation time (s)')
    ax.set_xlabel('Model')
    plt.savefig('./time_comparison.png', dpi=100)
    plt.show()
    
    plt.close()


def main():
    model_data_dict = speed_and_error_comparison(100)
    plot_model_comparison(model_data_dict)

if __name__ == "__main__":
    main()