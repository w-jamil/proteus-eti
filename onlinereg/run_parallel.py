from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import glob
import time
import random
import argparse
from collections import defaultdict

import ray
import numpy as np
import matplotlib.pyplot as plt

from coirr import COIRR
from dataset import Dataset

parser = argparse.ArgumentParser(description="Run the asynchronous parameter "
                                             "server example.")

parser.add_argument("--algorithm", choices=["coirr", "oslog", "onlmsr"],
                    help="The name of the algorithm", required=True)

parser.add_argument("--tuning-parameter", default=0.6, type=float,
                    help="The value of the tuning parameter.")

parser.add_argument("--data-file", type=str,
                    help="path of the data file", required=True)

parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")


def to_module(name):
    return '.'.join([name, name.upper()])


def get_class(path):
    from importlib import import_module
    module_path, _, class_name = path.rpartition('.')
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass


@ray.remote
class ParameterServer(object):
    """This class implements a parameter server which handles model params synchronization
    """
    def __init__(self, num_workers, dataset, algorithm, tuning_param, nb_features):
        self.workers_finished = [False] * num_workers
        self.model = get_class(to_module(algorithm))(tuning_param, nb_features) 
        self.dataset = dataset

    def push(self, updates):
        """To receive model updates from the workers
        """
        self.model.apply_updates(updates)

    def pull(self):
        """To send model parasm to the workers
        """
        return self.model.get_params()
    
    def update_finished(self, worker_id):
        """Worker informs the PS when its job is finished
        """
        self.workers_finished[worker_id-1] = True

    def finished_training(self):
        return all(self.workers_finished)
    
    def next_input(self):
        """Return next datapoint of the dataset
        """
        return self.dataset.next()
    
    def save_plot(self, expected, predicted):
        plt.scatter(expected, predicted)
    
    def show_plot(self):
        plt.show()
    

@ray.remote
def worker_task(worker_id, ps, algorithm, tuning_param, nb_features):
    """This function is executed as a remote task on workers  
    """
    # Initialize the model.
    model = get_class(to_module(algorithm))(tuning_param, nb_features)

    while True:
        # Get current params from the parameter server.
        # print('worker %d pull params'% (worker_id))
        params = ray.get(ps.pull.remote())

        # print('worker %d set params'% (worker_id))
        model.apply_updates(params)

        # print('worker %d next datapoint'% (worker_id))
        x_t, expected = ray.get(ps.next_input.remote())
        if len(x_t) > 0: 
            # always makes prediction before model updates
            predicted = model.predict(x_t)
            ps.save_plot.remote(expected, predicted)
            print('worker %d predicted %f expected %f'% (worker_id, predicted, expected))
            # update the model
            updates = model.delta(x_t, expected)
            ps.push.remote(updates)
        else:
            print('worker %d finished' %(worker_id))
            ps.update_finished.remote(worker_id)
            return


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_workers=args.num_workers, include_webui=False)

    with open(args.data_file) as f:
        first_line = f.readline()
        nb_features = len(first_line.split(' ')[:-1])
    
    dataset = Dataset(args.data_file)

    # create a parameter server with some random weights.
    ps = ParameterServer.remote(args.num_workers, dataset, args.algorithm, args.tuning_parameter, nb_features)

    # call remote tasks
    for i in range(args.num_workers):
        worker_task.remote(i + 1, ps, args.algorithm, args.tuning_parameter, nb_features)

    # wait until all remote tasks are finished
    finished = False
    while not finished:
        time.sleep(1)
        finished = ray.get(ps.finished_training.remote())
    
    ps.show_plot.remote()

    time.sleep(5)
    