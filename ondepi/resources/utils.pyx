import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import logging


def get_logger(str name="ondepi", level=logging.INFO):
    logger = logging.Logger("ondepi")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger

logger = get_logger()


def check_nonempty_df(df):
    if len(df)==0:
        logger.error("Dafatame has length zero")
        raise ValueError()

def timestamp_to_idx(
        np.ndarray[double, ndim=1] times):
    cdef int n = 9
    cdef np.ndarray[long, ndim=1] idx = np.array(
            np.floor(times * (10**n)), dtype=long)
    return idx

def idx_to_timestamp(
        np.ndarray[long, ndim=1] idx):
    cdef int n = 9
    cdef np.ndarray[double, ndim=1] idx_f = np.array(idx, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] times = np.array(idx_f * 10.0**(-n), dtype=np.float64)
    return times

def sample_to_arrays(Sample sample):
    cdef long unsigned int sample_size = sample.observations.size()
    cdef vector[double] times
    times.reserve(sample_size)
    cdef vector[EventType] events
    events.reserve(sample_size)
    cdef vector[long] states
    states.reserve(sample_size)
    cdef EventState es
    cdef long unsigned int n
    for n in range(sample_size):
        es = sample.observations.at(n)
        times.push_back(es.time)
        events.push_back(es.event)
        states.push_back(es.state)
    cdef np.ndarray[double, ndim=1] arr_times = np.array(times)    
    cdef np.ndarray[long, ndim=1] arr_events = np.array(events)    
    cdef np.ndarray[long, ndim=1] arr_states = np.array(states)    
    return arr_times, arr_events, arr_states

cpdef Sample arrays_to_sample(
        vector[double] times, 
        vector[EventType] events, 
        vector[long] states):
    cdef Sample sample
    cdef long unsigned int sample_size = times.size()
    sample.observations.reserve(sample_size)
    cdef EventState es
    cdef long unsigned int n
    for n in range(sample_size):
        es.time = times.at(n)
        es.state = states.at(n)
        es.event = events.at(n)
        sample.observations.push_back(es)
    return sample

def arrays_to_df(
    np.ndarray[double, ndim=1] times,
    np.ndarray[long, ndim=1] events,
    np.ndarray[long, ndim=1] states,
    ):
    cdef np.ndarray[long, ndim=1] N_D = np.array(np.cumsum(events == 0), dtype=np.int64)
    cdef np.ndarray[long, ndim=1] N_A = np.array(np.cumsum(events == 1), dtype=np.int64)
    cdef np.ndarray[long, ndim=1] time_i = timestamp_to_idx(times)
    df = pd.DataFrame({
        'time_i': time_i, 
        'time': times,
        'event': events,
        'state': states,
        'N_D': N_D,
        'N_A': N_A,
        })
    return df


def sample_to_df(Sample sample):
    return arrays_to_df(*sample_to_arrays(sample))


def launch_serial(fun, list_args):
    cdef list res = []
    for args in list_args:
        try:
            res.append(fun(*args))
        except Exception as e:
            print(type(e), e)
    return res        

def launch_async(fun, list_args):
    cdef int tot_tasks = len(list_args)
    cdef int num_processes = os.cpu_count()
    cdef int max_num_tasks = 1 + max(1, tot_tasks // num_processes)
    cdef list results = []
    cdef list async_res = []
    def store_res(res):
        try:
            results.append(res)
        except Exception as e:
            print("Storing result failed with exception {}".format(e))
    def error_handler(e):
        print("Error in worker: {}".format(e))
    with mp.Pool(processes=num_processes, maxtasksperchild=max_num_tasks) as pool:
        for args in list_args:
            async_res.append(
                    pool.apply_async(
                        fun,
                        args = args,
                        callback=store_res,
                        error_callback=error_handler
                        )    
                    )
        pool.close()
        pool.join()
        pool.terminate()
    cdef int i=0
    for i in range(len(async_res)):
        async_res[i].successful()
    return results    

def select_best_optimization_result(list_of_results):
    results = [res for res in list_of_results if res.get('success')]
    if results == []:
        raise ValueError("No minimisation succeded")
    funs = np.array([res.get('fun') for res in results], dtype=np.float64)
    idx_min = np.argmin(funs)
    return results[idx_min]


def extract_times_of_event(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        long event = 1
        ):
    cdef int n = 0
    idx = events==event
    cdef np.ndarray[double, ndim=1] res = times[idx]
    return res

def extract_states_at_event(
        np.ndarray[long, ndim=1] states,
        np.ndarray[long, ndim=1] events,
        long event = 1
        ):
    cdef int n = 0
    idx = events==event
    cdef np.ndarray[long, ndim=1] res = states[idx]
    return res

