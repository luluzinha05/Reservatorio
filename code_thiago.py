####################################################################
# Import libraries
####################################################################
import numpy as np
from mpi4py import MPI
from tools import input_simulation_parameters
from tools import DirichletBC3D
from simuladores.slab import slab3D
from tools import interactive_plot_3D
from tools import run_advection_solver_3D
from tools import interactive_solution_3D
from tools import InitialConditionOilStain
####################################################################

####################################################################
# Inicializacao MPI
####################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
####################################################################

####################################################################
# Rank 1: carrega dados, executa e envia
####################################################################
if rank == 1:

    internal_simulpar=input_simulation_parameters('simulation_input2.in')

    nx = internal_simulpar.mesh[0]
    ny = internal_simulpar.mesh[1]
    nz = internal_simulpar.mesh[2]

    Lx = internal_simulpar.Dom[0]
    Ly = internal_simulpar.Dom[1]
    Lz = internal_simulpar.Dom[2]

    Y = 10+0*np.random.rand(nx,ny,nz)

    internal_simulpar.BC = DirichletBC3D
    ####################################################################

    ####################################################################
    # Solve pressure and velocities
    ####################################################################
    coord, K, psim,dsim,vxsim,vysim, vzsim, divsim = slab3D(internal_simulpar,Y)

    #interactive_plot_3D(psim, K, Lx, Ly)
    ####################################################################

    ####################################################################
    # Solve transport
    ####################################################################
    cfl = 1.0
    day = 86400 # seconds in a day
    tf = 60*day # 2 months in seconds
    IC = InitialConditionOilStain(20, 80, 20, 80, 0, nz, Lx/nx, Ly/ny, Lz/nz)

    x, y, z, c_hist, dt, nt = run_advection_solver_3D(Lx, Ly, Lz, nx, ny, nz, 
                                                  ux = vxsim, uy = vysim, uz = vzsim, 
                                                  cfl = cfl, tf = tf, IC=IC)
    ####################################################################

    # Estrutura de dados 
    data_to_send = {
            "nx": nx, "ny": ny, "nz": nz,
            "Lx": Lx, "Ly": Ly, "Lz": Lz,
            "K": K, "dt": dt, "nt": nt,
            "vxsim": vxsim, "vysim": vysim, "vzsim": vzsim,
            "c_hist": c_hist,
            "IC": IC
    }

    comm.send(data_to_send, dest=0)

####################################################################
# Rank 0: recebe e mostra a solucao
####################################################################
elif rank == 0:
    data_received = comm.recv(source=1)
    print(">>> Dados recebidos do Rank 1 com sucesso!")

    interactive_solution_3D(
        data_received["c_hist"], data_received["K"], data_received["dt"],
        data_received["Lx"], data_received["Ly"], data_received["Lz"],
        data_received["vxsim"][0,0,0], data_received["vysim"][0,0,0], data_received["vzsim"][0,0,0],
        data_received["IC"]
    )
####################################################################
