import os
import multiprocessing as mp
import numpy as np

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


def generate_init_guesses(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        long event = 1,
        int num = 5
        ):
    cdef long unsigned int n = 0
    cdef np.ndarray[double, ndim=1] T_event = extract_times_of_event(times, events, event=event)
    cdef double interarrival_avg = np.mean(np.diff(times))
    cdef double interarrival_avg_event = np.mean(np.diff(T_event))
    cdef list alpha_1s = np.random.uniform(-10.0, 10.0, size=num).tolist()
    cdef list init_guesses = []
    cdef np.ndarray[double, ndim=1] param = np.ones(5, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] guess = np.ones(5, dtype=np.float64)
    for alpha_1 in alpha_1s:
        param[1] = alpha_1
        param[4] = interarrival_avg
        guess = np.array(param, copy=True)
        init_guesses.append(guess)
        param[4] = interarrival_avg_event
        guess = np.array(param, copy=True)
        init_guesses.append(guess)
    return init_guesses   

