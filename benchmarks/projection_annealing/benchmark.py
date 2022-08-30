import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import numpy as np
import sys
from argparse import ArgumentParser

sys.path.append('../..')  # PINNFramework etc.
import examples as pinn_examples
import PINNFramework as pf
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pde", dest="pde",default='maxwell',type=str)
    parser.add_argument("--annealing", dest="annealing", default=0, type=int)
    parser.add_argument("--projection", dest="projection", default=0, type=int)
    args = parser.parse_args()

    if args.pde == "schrodinger":
        input_dimension = 2
        output_dimension = 2
        lb = np.array([-5.0, 0.0])
        ub = np.array([5.0, np.pi / 2])
        nb = 50
        nf = 20000
        n0 = 50
        initial_condition = pf.InitialCondition(pinn_examples.Schroedinger1D.InitialConditionDataset(n0),
                                                args.pde + "ic")

        pde_loss = pf.PDELoss(pinn_examples.Schroedinger1D.PDEDataset(nf, lb, ub),
                              pinn_examples.Schroedinger1D.schroedinger1d,
                              args.pde + "pde")
        bc_dataset = pinn_examples.Schroedinger1D.BoundaryConditionDataset(nb,lb,ub)
        boundary_condition =  [pf.PeriodicBC(bc_dataset, 0, "u periodic boundary condition"),
                               pf.PeriodicBC(bc_dataset, 1, "v periodic boundary condition"),
                               pf.PeriodicBC(bc_dataset, 0, "u_x periodic boundary condition", 1, 0),
                               pf.PeriodicBC(bc_dataset, 1, "v_x periodic boundary condition", 1, 0)]
        model = pf.models.MLP(input_size=2,
                              output_size=2,
                              hidden_size=50,
                              num_hidden=4,
                              lb=lb,
                              ub=ub)

    if args.pde == "burgers":
        nf = 10000
        n0 = 100
        input_dimension = 2
        output_dimension = 1
        initial_condition = pf.InitialCondition(pinn_examples.BurgersEquation1D.InitialConditionDataset(n0),
                                                args.pde + "ic")
        pde_dataset = pinn_examples.BurgersEquation1D.PDEDataset(nf)
        pde_loss = pf.PDELoss(pde_dataset,
                              pinn_examples.BurgersEquation1D.burger1D,
                              args.pde + "pde")
        boundary_condition = []
        model = pf.models.MLP(input_size=2, output_size=1,
                              hidden_size=40, num_hidden=8, lb=pde_dataset.lb, ub=pde_dataset.ub, activation=torch.tanh)

    if args.pde == "heat":
        lb = np.array([0, 0.0])
        ub = np.array([1.0, 2.0])
        nb = 50
        nf = 10000
        n0 = 50
        input_dimension = 2
        output_dimension = 1

        initial_condition = pf.InitialCondition(pinn_examples.HeatEquation1D.InitialConditionDataset(n0),
                                                args.pde + "ic")
        pinn_examples.HeatEquation1D.PDEDataset(nf, lb, ub)
        pde_loss = pf.PDELoss(pinn_examples.HeatEquation1D.PDEDataset(nf, lb, ub),
                              pinn_examples.HeatEquation1D.heat1d,
                              args.pde + "pde")
        boundary_condition = [
        pf.DirichletBC(pinn_examples.HeatEquation1D.func,
                       pinn_examples.HeatEquation1D.BoundaryConditionDatasetlb(nb, lb, ub), name='bc lb'),
        pf.DirichletBC(pinn_examples.HeatEquation1D.func,
                       pinn_examples.HeatEquation1D.BoundaryConditionDatasetub(nb, lb, ub), name='bc ub')
        ]

        model = pf.models.MLP(input_size=2,
                              output_size=1,
                              hidden_size=50,
                              num_hidden=4,
                              lb=lb,
                              ub=ub)


    if args.pde == "maxwell":
        lb = np.array([0, 0.0])
        ub = np.array([4.0, 5.0])
        nb = 50
        nf = 10000
        n0 = 50
        input_dimension = 2
        output_dimension = 2
        initial_condition = pf.InitialCondition(pinn_examples.Maxwell1D.InitialConditionDataset(n0),
                                                args.pde + "ic")

        pde_loss = pf.PDELoss(pinn_examples.Maxwell1D.PDEDataset(nf, lb, ub),
                              pinn_examples.Maxwell1D.maxwell_1d,
                              args.pde + "pde")
        boundary_condition = []
        model = pf.models.FingerNet(lb, ub, 2, 2, 50, 3, 5, torch.sin, False)


    pinn = pf.PINN(model,input_dimension,output_dimension,pde_loss, initial_condition, boundary_condition,use_gpu=True)
    logger = pf.WandbLogger("Projection benchmark", args, 'aipp')
    pinn.fit(50000,
             pinn_path=args.pde + '_best_model_projection_{}_annealing_{}.pt'.format(args.projection,args.annealing),
             writing_cycle=1,
             logger=logger,
             track_gradient=True,
             activate_annealing=args.annealing,
             annealing_cycle=10,
             gradient_projection=args.projection)








