# Galileo23-MC1-GPU

[**Solid Earth and Geohazards in the Exascale Era** | BARCELONA | SPAIN | 23–26 MAY 2023](https://egu-galileo.eu/gc11-solidearth/)

#### Master class on GPU HPC for the exascale era in geosciences

_by Ludovic Räss and Ivan Utkin (ETH Zurich)_
## Program
| Slot    | Program |
| :-----  | :----- |
| [Slot 1](#slot-1)  | **Introduction** about GPU HPC<br>- Student short presentation (5 min each) <br>- Getting set-up with GPU node access on `octopus`<br>- Brief intro about Julia for HPC |
| [Slot 2](#slot-2) | **Hands-on I**<br>- GPU computing and HPC<br>- Julia GPU and MPI stack<br>- Model design and implementation: Stokes flow in a channel |
| [Slot 3](#slot-3) | **Hands-on II**<br>- Mulit-GPU computing<br>- Performance limiters<br>- Performance measure $T_\mathrm{eff}$ |
| [Slot 4](#slot-4) | **OPTION 1 - Towards scalable inverse modelling**<br>- AD (GPU) tools in Julia<br>- Jacbian-vector product (JVP) computation in a multi-GPU model<br>- _Advanced: towards sensitivity kernels and adjoint solutions_<br> **Wrap-up discussion** |

## Content
- [Getting started](#getting-started)
- [Slot 1 - Intro](#slot-1)
- [Slot 2 - Hands-on I](#slot-2)
- [Slot 3 - Hands-on II](#slot-3)
- [Slot 4 - OPTION 1](#slot-4)

## Getting started
This section provides directions on getting your GPU HPC dev environment ready on the `octopus` supercomputer at the University of Lausanne, Switzerland. During tthis Master-class, we will use SSH to login to a remote multi-GPU compute node on `octopus`. Each of the participant should get access to 4 Nvidia Titan Xm 12GB. 

In the following, we will give directions on how to use [VSCode](https://code.visualstudio.com) and the [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension to access the compute resources. However, feel free to access the resources using your preferred SSH setup.

1. Download [VSCode](https://code.visualstudio.com/download) on your laptop.
2. Install the [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) extensions.
3. Retrieve your **confidential** login credentials from the email you received titled "MC1 login credentials".
4. Setup a password-less SSH config to access `octopus`. See e.g. [here on "how-to"]().
5. [Edit the SSH config file]() to add the infos about `octopus` login (replacing `XX` with the number you got assigned - note the node number should be `node0X` for `X<10`):
    ```
    Host octo-login
      HostName achsrv0.unil.ch
      User courseXX
      IdentityFile ~/.ssh/id_ed25519

    Host nodeXX
      HostName nodeXX.octopoda
      User courseXX
      ProxyJump octo-login
    ```
6. Connect to your assigned node, check you are in your home folder (using `pwd`) and clone this repo into your home:
    ```
    git clone https://github.com/PTsolvers/Galileo23-MC1-GPU.git
    ```
7. Load the Julia, CUDA and MPI modules:
    ```
    module load julia cuda/11.4 openmpi/gcc83-314-c112
    ```
    and launch Julia typing `julia`.
8. In Julia, type `]` to enter the "package-mode". There, activate the current project and resolve the packages we will need typing:
    ```julia-repl
    (@v1.9) pkg> activate .
    
    (@Galileo23-MC1-GPU) pkg> resolve
    ```
9. To make sure you are all set, check your CUDA and MPI install:
    ```julia-repl
    julia> using CUDA

    julia> CUDA.versioninfo()
    CUDA runtime 11.2, local installation
    CUDA driver 12.1
    NVIDIA driver 530.30.2
    
    # [skipped lines]
    
    4 devices:
      0: NVIDIA GeForce GTX TITAN X (sm_52, 11.918 GiB / 12.000 GiB available)
      1: NVIDIA GeForce GTX TITAN X (sm_52, 11.918 GiB / 12.000 GiB available)
      2: NVIDIA GeForce GTX TITAN X (sm_52, 11.918 GiB / 12.000 GiB available)
      3: NVIDIA GeForce GTX TITAN X (sm_52, 11.918 GiB / 12.000 GiB available)
    
    julia> MPI.MPI_LIBRARY_VERSION_STRING
    "Open MPI v3.1.4, package: Open MPI root@node01.octopoda Distribution, ident: 3.1.4, repo rev: v3.1.4, Apr 15, 2019\0"
    ```
10. Finally, you should be able to run the following scripts to make sure MPI-based GPU selection and GPU-aware MPI is running as expected in Julia. Exit Julia and go to the `scripts_start` folder:
    ```
    cd scripts_start
    ```
    Then run the [`hello_mpi_gpu.jl`](scripts_start/hello_mpi_gpu.jl) script to make sure GPU selection works as expected:
    ```
    mpirun -np 4 -mca btl_openib_warn_default_gid_prefix 0 julia --project hello_mpi_gpu.jl
    ```
    and the [`alltoall_mpi_gpu.jl`](scripts_start/alltoall_mpi_gpu.jl) script to verify GPU-aware MPI is working:
    ```
    mpirun -np 4 -mca btl_openib_warn_default_gid_prefix 0 julia --project alltoall_mpi_gpu.jl
    ```
## Slot 1

## Slot 2

## Slot 3

## Slot 4