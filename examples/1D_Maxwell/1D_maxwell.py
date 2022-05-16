import numpy as np
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.constants as constants
from pyDOE import lhs
from torch.autograd import grad
from torch import ones, stack, Tensor
from torch.utils.data import Dataset
from argparse import ArgumentParser

sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf


class PDEDataset(Dataset):
    def __init__(self, nf, lb, ub):
        self.xf = lb + (ub - lb) * lhs(2, nf)

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        return Tensor(self.xf).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1


class InitialConditionDataset(Dataset):

    def __init__(self, n0, iteration=5):
        """
        Constructor of the boundary condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        e_x_data = np.load("EX.npy")
        h_y_data = np.load("HY.npy")
        exact_e = e_x_data[iteration, :]  # load timestep
        exact_h = h_y_data[iteration, :]  # load timestep
        exact_e = exact_e.reshape(-1, 1)
        exact_h = exact_h.reshape(-1, 1)
        print(exact_e.shape)
        x = np.arange(0, e_x_data.shape[1])
        x = x.reshape(-1, 1)

        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x, :]
        self.e = exact_e[idx_x, 0:1]
        self.h = exact_h[idx_x, 0:1]
        self.t = np.zeros(self.x.shape)

    class BoundaryConditionDataset(Dataset):

        def __init__(self, nb, lb, ub):
            """
            Constructor of the boundary condition dataset

            Args:
              nb (int)
            """
            super(type(self)).__init__()
            t = np.arange(0,100)

            x_lb = np.concatenate((0 * t + lb[0], t), 1)  # (lb[0], tb)
            x_ub = np.concatenate((0 * t + ub[0], t), 1)  # (ub[0], tb)
            idx_t = np.random.choice(t.shape[0], nb, replace=False)
            x_boundary = np.vstack([x_lb, x_ub])
            self.x = x_boundary[idx_t, :]

            def __getitem__(self, idx):
                """
                Returns data for initial state
                """
                return Tensor(self.x_lb).float(), Tensor(self.x_ub).float()


    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = np.concatenate([self.e, self.h], axis=1)
        return Tensor(x).float(), Tensor(y).float()

def maxwell_1d(x, u):
    # u predicts electric field and magnetic field
    # x is a tensor x,U

    # faraday law = nabla x E = dB/dt
    # ampere law = nabla x B = 1/c dE/dt  --> current is equal to zero

    pred = u
    e = pred[:, 0]
    h = pred[:, 1]

    grads = ones(e.shape, device=u.device)  # move to the same device as prediction
    grad_e = grad(e, x, create_graph=True, grad_outputs=grads)[0]
    grad_h = grad(h, x, create_graph=True, grad_outputs=grads)[0]

    e_z = grad_e[:, 0]
    e_t = grad_e[:, 1]

    h_z = grad_h[:, 0]
    h_t = grad_h[:, 1]

    f_e = e_t - h_z
    f_h = h_t - e_z
    return stack([f_e, f_h], 1)

if __name__ == "__main__":
    lb = np.array([0, 0])
    ub = np.array([40, 100])
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=50000, help='Number of training iterations')
    parser.add_argument('--n0', dest='n0', type=int, default=40, help='Number of input points for initial condition')
    parser.add_argument('--nf', dest='nf', type=int, default=10000, help='Number of input points for pde loss')
    parser.add_argument('--num_hidden', dest='num_hidden', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100, help='Size of hidden layers')
    args = parser.parse_args()
    ic_dataset = InitialConditionDataset(args.n0, 30)
    initial_condition = pf.InitialCondition(ic_dataset, name='Initial Condition')

    pde_dataset = PDEDataset(args.nf, lb, ub)
    pde_loss = pf.PDELoss(pde_dataset, maxwell_1d, name='Maxwell equation')

    model = pf.models.MLP(input_size=2,
                          output_size=2,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=lb,
                          ub=ub)
    pinn = pf.PINN(model, 2, 2, pde_loss, initial_condition, [], use_gpu=True)
    """"
    logger = pf.WandbLogger('1D Maxwell equation', args, 'aipp')

    pinn.fit(args.num_epochs, checkpoint_path='checkpoint.pt',
             restart=True, logger=logger, activate_annealing=False, annealing_cycle=None,
             writing_cycle=500,
             track_gradient=False)
    """
    pinn.load_model('best_model_pinn.pt')
    # plotting
    x = np.arange(0, 40)
    t = np.arange(0, 100)

    X, T = np.meshgrid(x, t, indexing='ij')
    inp_X = X.reshape(-1, 1)
    inp_T = T.reshape(-1, 1)

    test_input = np.hstack([inp_X, inp_T])
    pred = model(Tensor(test_input).cuda())
    pred_e = pred[:, 0].detach().cpu().numpy()
    pred_h = pred[:, 1].detach().cpu().numpy()
    pred_e = pred_e.reshape(X.shape)
    pred_h = pred_h.reshape(X.shape)
    print(pred_e.shape)
    plt.pcolormesh(t, x, pred_e)
    plt.title('Electric field')
    plt.colorbar()
    plt.show()
    plt.pcolormesh(t, x, pred_h)
    plt.title("Magnetic field")
    plt.colorbar()
    plt.show()
    pred_e = pred_e.T
    pred_h = pred_h.T
    gt_e = np.load("EX.npy")
    gt_h = np.load("HY.npy")
    # timestep 0
    plt.title('e_field time step 0')
    plt.plot(gt_e[10, :], label='ground truth')
    plt.plot(pred_e[0, :], label='prediction')
    plt.legend()
    plt.show()
    # timestep 50
    plt.title('e_field time step 10')
    plt.plot(gt_e[20, :], label='ground truth')
    plt.plot(pred_e[10, :], label='prediction')
    plt.legend()
    plt.show()

    # timestep 75
    plt.title('e_field time step 15')
    plt.plot(gt_e[25,:], label='ground truth')
    plt.plot(pred_e[15, :], label='prediction')
    plt.legend()
    plt.show()


    # timestep 0
    plt.title('b_field time step 0')
    plt.plot(gt_h[10,:], label='ground truth')
    plt.plot(pred_h[0, :], label='prediction')
    plt.legend()
    plt.show()
    # timestep 50
    plt.title('b_field time step 10')
    plt.plot(gt_h[20,:], label='ground truth')
    plt.plot(pred_h[15, :], label='prediction')
    plt.legend()
    plt.show()

    # timestep 75
    plt.title('b_field time step 15')
    plt.plot(gt_h[25,:], label='ground truth')
    plt.plot(pred_h[15, :], label='prediction')
    plt.legend()
    plt.show()
