import numpy as np
import torch
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
from pyDOE import lhs
from torch.autograd import grad
from torch import ones, stack, Tensor
from torch.utils.data import Dataset
from argparse import ArgumentParser

sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf

def gaussian_pulse(x,mu=2):
    return np.exp((-(x-mu)**2)/2) * np.sin(2*np.pi * 4 * (x-mu))

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

    def __init__(self, n0):
        """
        Constructor of the boundary condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        x = np.linspace(0, 4, n0)

        self.exact_e = gaussian_pulse(x)
        self.exact_h = gaussian_pulse(x)
        exact_e = self.exact_e.reshape(-1, 1)
        exact_h = self.exact_h.reshape(-1, 1)

        x = x.reshape(-1, 1)
        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x, :]
        self.e = exact_e[idx_x, 0:1]
        self.h = exact_h[idx_x, 0:1]
        self.t = np.zeros(self.x.shape)

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
    ub = np.array([4, 5])
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=20000, help='Number of training iterations')
    parser.add_argument('--n0', dest='n0', type=int, default=300, help='Number of input points for initial condition')
    parser.add_argument('--nf', dest='nf', type=int, default=50000, help='Number of input points for pde loss')
    parser.add_argument('--num_hidden', dest='num_hidden', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100, help='Size of hidden layers')
    parser.add_argument('--annealing', dest='annealing', type=int, default=0, help='Activate Annealing')
    parser.add_argument('--pretraining',dest='pretraining', type=int, default=0, help='Activate pretraining')
    parser.add_argument('--projection', dest='projection', type=int, default=1, help='Activate projection')
    parser.add_argument('--ic_weight', dest='ic_weight', type=int, default=1)
    args = parser.parse_args()
    ic_dataset = InitialConditionDataset(args.n0)
    initial_condition = pf.InitialCondition(ic_dataset, name='Initial Condition')

    pde_dataset = PDEDataset(args.nf, lb, ub)
    pde_loss = pf.PDELoss(pde_dataset, maxwell_1d, name='Maxwell equation')

    model = pf.models.FingerNet(lb, ub, 2, 2, 50, 3, 5, torch.sin, False)
    #model = pf.models.MLP(2,2, args.hidden_size,args.num_hidden, lb, ub)
    pinn = pf.PINN(model, 2, 2, pde_loss, initial_condition, [], use_gpu=True)
    run_name = input('Enter wandb run name: ')
    logger = pf.WandbLogger('Projection Exeperiments', args, run_name=run_name)
    # logger = None

    # load best pre-training model
    pinn.load_model('pretraining_best_model_pinn.pt')

    pinn.fit(args.num_epochs, checkpoint_path='checkpoint.pt', epochs_pt=30000, pretraining=args.pretraining,
             restart=True, logger=logger, activate_annealing=args.annealing, annealing_cycle=200,
             writing_cycle=50, learning_rate=1e-3,  track_gradient=True,
             lbfgs_finetuning=False, gradient_projection=args.projection)


    ## Plotting
    # field plots
    x = np.linspace(0, 4, args.n0)
    t = np.linspace(0, 5, 200)

    T, X = np.meshgrid(t, x, indexing='ij')
    inp_X = X.reshape(-1, 1)
    inp_T = T.reshape(-1, 1)

    test_input = np.hstack([inp_X, inp_T])
    pred = model(Tensor(test_input).cuda())
    pred_e = pred[:, 0].detach().cpu().numpy()
    pred_h = pred[:, 1].detach().cpu().numpy()
    pred_e = pred_e.reshape(X.shape)
    pred_h = pred_h.reshape(X.shape)
    print(pred_e.shape)
    fig = plt.figure()
    plt.pcolormesh(t, x, pred_e.T)
    plt.title('Electric field')
    plt.colorbar()
    if logger is not None:
        plt.savefig(logger.name + 'e_field.png')
    plt.show()
    fig = plt.figure()
    plt.pcolormesh(t, x, pred_h.T)
    plt.title("Magnetic field")
    plt.colorbar()
    if logger is not None:
        plt.savefig(logger.name + 'b_field.png')
    plt.show()
    print(pred_e.shape)

    # initial condition plots
    fig = plt.figure()
    plt.title("Initial Condition E-Field ")
    plt.plot(x, pred_e[0, :], label='prediction 0')
    plt.plot(x, ic_dataset.exact_e, label='ground truth initial state')
    if logger is not None:
        plt.savefig(logger.name + 'initial_condition_e.png')
    plt.show()
    plt.figure()
    plt.title("Initial Conditioni H-Field")
    plt.plot(x, pred_h[0, :], label='prediction 0')
    plt.plot(x, ic_dataset.exact_h, label='ground truth initial state')
    if logger is not None:
        plt.savefig(logger.name + 'initial_condition_h.png')
    plt.show()

    # snapshot at different time-steps
    # timestep 0
    fig = plt.figure()
    plt.title('e_field time step')
    plt.plot(x, pred_e[0, :], label='prediction 0')
    plt.plot(x, ic_dataset.exact_e, label='ground truth initial state')
    # timestep 2
    plt.plot(x, pred_e[2, :], label='prediction 2')

    # timestep 5
    plt.plot(x, pred_e[4, :], label='prediction 5')
    plt.legend()
    if logger is not None:
        plt.savefig(logger.name + 'e_field_time.png')
    plt.show()

    # timestep
    plt.title('b_field')
    plt.plot(x, pred_h[0, :], label='prediction 0')

    # timestep 20
    plt.plot(x, pred_h[2, :], label='prediction 2')

    # timestep 75
    plt.plot(x, pred_h[4, :], label='prediction 5')
    plt.legend()
    if logger is not None:
        plt.savefig(logger.name + 'h_field_time.png')
    plt.show()