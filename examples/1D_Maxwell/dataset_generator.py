import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
BASE_PATH = "/home/patrick/Projekte/fdtd3d/relaese/Source/"
EX_PATH = "previous-1_[timestep={}]_[pid=0]_[name=Ex].txt"
HY_PATH = "previous-1_[timestep={}]_[pid=0]_[name=Hy].txt"


def load_field(base_path, path, timestep):
    return np.loadtxt(base_path+path.format(timestep))


if __name__ == "__main__":
    E_X = []
    grid = []
    for i in tqdm(range(1, 1001, 1)):
        current_state = load_field(BASE_PATH, EX_PATH, i) # (x, e(x))
        e_x = current_state[:, 1]
        E_X.append(e_x)

    E_X = np.array(E_X)
    H_Y = []
    for i in tqdm(range(1, 1001, 1)):
        current_state = load_field(BASE_PATH, HY_PATH, i)  # (x, e(x))
        hy = current_state[:, 1]
        H_Y.append(hy)

    E_X = np.array(E_X)
    H_Y = np.array(H_Y)



    plt.pcolormesh(E_X.T)
    plt.title('electric field')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

    plt.pcolormesh(H_Y.T)
    plt.title('magnetic field')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

    # saving
    np.save('EX.npy', E_X)
    np.save('HY.npy', H_Y)



