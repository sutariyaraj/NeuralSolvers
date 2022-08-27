import numpy as np
import torch
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.constants as constants
from pyDOE import lhs
from torch.autograd import grad
from torch import ones, stack, Tensor
from torch.utils.data import Dataset
from argparse import ArgumentParser
import importlib

sys.path.append('../..')  # PINNFramework etc.
import examples as pinn_examples
import PINNFramework as pf



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pde", dest="pde", type=str)
    parser.add_argument("--annealing", dest="annealing", type=int)
    parser.add_argument("--projection", dest="projection", type=int)
    args = parser.parse_args()

    if args.pde == "schrodinger":
        initial_condition = pf.InitialCondition(pinn_examples.Schroedinger1D.InitialConditionDataset(),
                                                args.pde + "ic")

        pde_loss = pf.PDELoss(pinn_examples.Schroedinger1D.PDEDataset(),
                              pinn_examples.Schroedinger1D.schroedinger1d,
                              args.pde + "pde")
        boundary_condition = pf.PeriodicBC(pinn_examples.Schroedinger1D.BoundaryConditionDataset(),2,args.pde + "boundary")


    if args.pde == "burgers":
        initial_condition = pf.InitialCondition(pinn_examples.BurgersEquation1D.InitialConditionDataset(),
                                                args.pde + "ic")

        pde_loss = pf.PDELoss(pinn_examples.BurgersEquation1D.PDEDataset(),
                              pinn_examples.BurgersEquation1D.burger1D,
                              args.pde + "pde")
        boundary_condition = []

    if args.pde == "heat":
        initial_condition = pf.InitialCondition(pinn_examples.HeatEquation1D.InitialConditionDataset(),
                                                args.pde + "ic")
        pinn_examples.HeatEquation1D.PDEDataset()
        pde_loss = pf.PDELoss(pinn_examples.HeatEquation1D.PDEDataset(),
                              pinn_examples.HeatEquation1D.heat1d,
                              args.pde + "pde")
        boundary_condition = [
        pf.DirichletBC(pinn_examples.HeatEquation1D.func,
                       pinn_examples.HeatEquation1D.BoundaryConditionDatasetlb()),
        pf.DirichletBC(pinn_examples.HeatEquation1D.func,
                       pinn_examples.HeatEquation1D.BoundaryConditionDatasetub())
        ]


    if args.pde == "maxwell":
        initial_condition = pf.InitialCondition(pinn_examples.Maxwell1D.InitialConditionDataset(),
                                                args.pde + "ic")

        pde_loss = pf.PDELoss(pinn_examples.Maxwell1D.PDEDataset(),
                              pinn_examples.Maxwell1D.maxwell_1d,
                              args.pde + "pde")
        boundary_condition = []






