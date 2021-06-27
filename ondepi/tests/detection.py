import numpy as np
import pandas as pd
from ondepi.resources.queue.queue import Queue, produce_df_detection


def main():
    queue = Queue()
    queue._set_param(
        nu_D_0=5.0,
        nu_D_1=-1.0,
        nu_D_2=2.0,
        alpha_D=3.0,
        beta_D=3.0,
        nu_A_0=10.0,
        nu_A_1=0.0,
        nu_A_2=1.0,
        alpha_A=2.0,
        beta_A=3.5
    )
    queue.simulate(
        max_time=1000.0,
        max_events=50,
        first_event=1,
        first_state=10
    )

    queue.filter(
        dt=0.00001,
        num_states=150
    )
    df = produce_df_detection(queue, beta=5.0)
    return df


if __name__ == '__main__':
    df = main()
    print(df)
