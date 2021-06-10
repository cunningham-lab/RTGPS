import pickle
import math
import logging
import os
import re
from datetime import datetime
from random import randint
import numpy as np


def get_vars_from_pkl(input_file):
    results = {}
    data = load_results(input_file)
    results.update({'ls': data['ls'], 'os': data['os'],
                    'noise': data['noise'], 'loss': data['loss']})
    return results


def get_vars_from_logs(input_file):
    results = {}
    ls = get_variable_np_array_from_log_file(variable_name='ls',
                                             path_to_file=input_file)
    noise = get_variable_np_array_from_log_file(variable_name='noise',
                                                path_to_file=input_file)
    output_scale = get_variable_np_array_from_log_file(variable_name='os',
                                                       path_to_file=input_file)
    loss = get_variable_np_array_from_log_file(variable_name='loss',
                                               path_to_file=input_file)
    results.update({'ls': ls, 'os': output_scale, 'noise': noise, 'loss': loss})
    return results


class TestMetrics:

    def __init__(self):
        self.db = {}
        self.datasets = {}

    def populate_db(self, lookup_path, key, translator):
        criteria = get_conditions(translator)
        logs = get_available_logs(lookup_path)
        for _, dataset_name in self.datasets.items():
            for log in logs:
                if log.find(dataset_name) > 0:
                    with open(file=lookup_path + log, mode='r') as f:
                        lines = f.readlines()
                        type_case = find_type_of_log(lines, translator, criteria)
                        rmse, nll = get_rmse_and_nll(lines)
                        seed = get_seed_from_log(lines)
                        self.db[key][type_case][dataset_name]['RMSE'].append(rmse)
                        self.db[key][type_case][dataset_name]['NLL'].append(nll)
                        self.db[key][type_case][dataset_name]['seed'].append(seed)

    def convert_to_np(self, consider_only=0):
        for model, output in self.db.items():
            for type_case, datasets in output.items():
                for dataset_name, metrics in datasets.items():
                    for metric, _ in metrics.items():
                        aux = self.db[model][type_case][dataset_name][metric]
                        out = np.array(aux)
                        out.sort()
                        if consider_only > 0:
                            out = out[0:consider_only]
                        self.db[model][type_case][dataset_name][metric] = out

    def create_db(self, model_name, types, metrics, datasets):
        metrics += ['seed']
        self.db = {model_name: {}}
        self.datasets = datasets
        for _, v in self.db.items():
            for t in range(1, types + 1):
                name = 'type_' + str(t)
                dataset_results = {}
                for _, dataset_name in self.datasets.items():
                    dataset_metrics = {}
                    for metric in metrics:
                        dataset_metrics.update({metric: []})
                    dataset_results.update({dataset_name: dataset_metrics})
                v.update({name: dataset_results})

    def load_db(self, input_file):
        self.db = load_results(input_file)

    def save_db(self, output_file):
        save_results(self.db, output_file)


def get_conditions(translator):
    criteria = {}
    for _, cases in translator.items():
        i = 0
        for k, _ in cases.items():
            i += 1
            criteria.update({i: k})
        break
    return criteria


def find_type_of_log(lines, translator, criteria):
    found_criteria = get_criteria_from_log(lines, criteria)
    for key, variables in translator.items():
        are_equal_dicts = variables == found_criteria
        if are_equal_dicts:
            type_case = key
    try:
        type_case
    except NameError:
        type_case = 0
    return type_case


def get_seed_from_log(lines):
    for line in lines:
        idx = line.find('seed')
        if idx > 0:
            seed = get_var_value(line, idx, 'seed')
    return seed


def get_criteria_from_log(lines, criteria):
    found = {}
    for line in lines:
        for _, var in criteria.items():
            idx = line.find(var)
            if idx > 0:
                retrived_value = get_var_value(line, idx, var)
                found.update({var: retrived_value})
    return found


def get_var_value(line, idx, var):
    subset = line[idx:]
    ss = subset.split(' ')
    ss = ss[1].rstrip().replace(',', '').replace('}', '')
    value = int(ss) if ss.isdigit() else ss
    if var in ['lr', 'coeff']:
        digits = re.compile(r'\-*[0-9]+.[0-9]+')
        value = float(digits.findall(value)[0])
    return value


def get_rmse_and_nll(lines):
    for line in lines:
        idx = line.find('Test RMSE')
        line_contains_metrics = line.find('Test RMSE') > 0
        if line_contains_metrics:
            subset = line[idx:]
            results = subset.split('|')
            for res in results:
                digits = re.compile(r'\-*[0-9]+.[0-9]+')
                out = float(digits.findall(res)[0])
                if res.find('RMSE') > 0:
                    rmse = out
                else:
                    nll = out
    return rmse, nll


def get_available_logs(path):
    logs = []
    for d in os.listdir(path):
        if not os.path.isdir(os.path.join(path, d)):
            if not d[-4:] == '.pkl':
                logs.append(d)
    return logs


def get_hypers(path_to_file):
    variables = ['model_name', 'rff_samples', 'total_iters', 'seed',
                 'min_val', 'max_val', 'truncation_name',
                 'warmup', 'coeff', 'step', 'optimizer', 'lr:']
    hyper_dict = {}
    with open(file=path_to_file, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find('Hyper') > 0:
                for var in variables:
                    idx = line.find(var)
                    if idx > 0:
                        subset = line[idx:]
                        ss = subset.split(' ')
                        ss = ss[1].rstrip().replace(',', '').replace('}', '')
                        ss = int(ss) if ss.isdigit() else ss
                        hyper_dict.update({var: ss})
    return hyper_dict


def get_variable_np_array_from_log_file(variable_name: str, path_to_file: str):
    variable_results = []
    with open(file=path_to_file, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            split = line.split(sep='|')
            if len(split) > 1:
                for part in split:
                    regex = re.compile(r'' + variable_name + ':')
                    is_variable_there = len(regex.findall(part)) > 0
                    if is_variable_there:
                        var = float(part.split()[1])
                        variable_results.append(var)
        variable_np = np.array(variable_results)
    return variable_np


def print_time_taken(tic, toc, text='Experiment took', logger=None):
    minutes = math.floor((toc - tic) / 60)
    seconds = (toc - tic) - minutes * 60
    message = text + f': {minutes:4d} min and {seconds:4.2f} sec'
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def load_results(input_file):
    with open(input_file, mode='rb') as f:
        results = pickle.load(f)
    return results


def save_results(results, output_file):
    with open(file=output_file, mode='wb') as f:
        pickle.dump(obj=results, file=f)


def start_all_logging_instruments(settings, results_path):
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    log_file_name = results_path + settings['dataset_name']
    log_file_name += '_' + time_stamp
    log_file_name += '_' + str(randint(1, 100000)) + '.log'
    logger_name = 'log_' + time_stamp + '_' + str(randint(1, 100000))
    logger = setup_logger(log_file_name, logger_name)
    log_all_settings(settings, logger)
    logger.log_file_name = log_file_name
    return logger


def log_all_settings(settings, logger):
    for key, value in settings.items():
        if key == 'ls':
            if len(value.shape) == 0:
                logger.info(f'Hyper: {key}: {value}')
            else:
                logger.info(f'Hyper: {key}: {value[0]}')
        else:
            logger.info(f'Hyper: {key}: {value}')


def setup_logger(log_file_name, logger_name: str = None):
    if logger_name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:    %(message)s')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(filename=log_file_name)
    file_handler.setFormatter(fmt=formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=stream_formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger
