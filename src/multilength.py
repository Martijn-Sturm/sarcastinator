import numpy as np 

def get_x_y(max_time_steps):
    for k in range(int(1e3)):
        time_steps = np.random.choice(range(1, max_time_steps), size=1)[0]
        if k % 2 == 0:
            x_train = np.expand_dims([np.insert(np.zeros(shape=(time_steps, 1)), 0, 1)], axis=-1)
            y_train = [1]
        else:
            x_train = np.array([np.zeros(shape=(time_steps, 1))])
            y_train = [0]
        print('\nInput: sequence of length {}\n'.format(time_steps))
        # print(x_train.shape)
        print(np.expand_dims(y_train, axis=-1))
        # yield x_train, np.expand_dims(y_train, axis=-1)

a = get_x_y(max_time_steps= 100)