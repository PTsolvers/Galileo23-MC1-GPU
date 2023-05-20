# Galileo23 Master-class 1 - GPU HPC in Julia

[**Solid Earth and Geohazards in the Exascale Era** | BARCELONA | SPAIN | 23–26 MAY 2023](https://egu-galileo.eu/gc11-solidearth/)

#### Master class on GPU HPC for the exascale era in geosciences

_by Ludovic Räss and Ivan Utkin (ETH Zurich)_
## Program
| Slot    | Program |
| :-----  | :----- |
| [Slot 1](#slot-1)  | **Introduction** about GPU HPC<br>- Student short presentation (5 min each) <br>- [Getting started](#getting-started) with GPU node access on `octopus`<br>- Brief intro about Julia for HPC |
| [Slot 2](#slot-2) | **Hands-on I**<br>- GPU computing and HPC<br>- Julia GPU and MPI stack<br>- Model design and implementation: Stokes flow in a channel |
| [Slot 3](#slot-3) | **Hands-on II**<br>- Multi-GPU computing<br>- Performance limiters<br>- Performance measure $T_\mathrm{eff}$ |
| [Slot 4](#slot-4) | **OPTION 1 - Towards scalable inverse modelling**<br>- AD (GPU) tools in Julia<br>- Jacbian-vector product (JVP) computation in a multi-GPU model<br><br>- _Advanced 1: towards sensitivity kernels and adjoint solutions_<br>- _Advanced 2: the accelerated pseudo-transient method_<br><br>  **Wrap-up discussion** |

## Content
- [Slot 1 - Intro](#slot-1)
- [Slot 2 - Hands-on I](#slot-2)
- [Slot 3 - Hands-on II](#slot-3)
- [Slot 4 - OPTION 1](#slot-4)

## Slot 1
### Getting started
This section provides directions on getting your GPU HPC dev environment ready on the `octopus` supercomputer at the University of Lausanne, Switzerland. During this Master-class, we will use SSH to login to a remote multi-GPU compute node on `octopus`. Each of the participant should get access to 4 Nvidia Titan Xm 12GB. 

> ⚠️ It is warmly recommended trying to perform the Getting started steps before the beginning of the workshop.

<details>
<summary>CLICK HERE for the getting started steps 🚀</summary>
<br>

In the following, we will give directions on how to use [VSCode](https://code.visualstudio.com) and the [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension to access the compute resources. However, feel free to access the resources using your preferred SSH setup.

1. Download [VSCode](https://code.visualstudio.com/download) on your laptop.
2. Install the [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) extensions.
3. Retrieve your **confidential** login credentials from the email you received titled "MC1 login credentials", namely your username `<username>` (in the format `courseXX`) and dedicated compute node ID `<nodeID>` (in the format `nodeXX`).
4. Setup a password-less SSH config to access `octopus` (see e.g. [here](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/) on "how-to"). Ideally, use `ed25519` encryption.
5. [Edit the SSH config file](https://code.visualstudio.com/blogs/2019/10/03/remote-ssh-tips-and-tricks#_ssh-configuration-file) to add the infos about `octopus` login (replacing `<username>` with the username you got assigned - note the node ID should always be a 2 digit number):
    ```
    Host octo-login
      HostName achsrv0.unil.ch
      User <username>
      IdentityFile ~/.ssh/id_ed25519

    Host node<nodeID>
      HostName node<nodeID>.octopoda
      User <username>
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
    
    (@Galileo23-MC1-GPU) pkg> instantiate
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
10. Let's try now to run some basic plotting scripts within Julia and get the output inlined to VSCode. In the [scripts_start](scripts_start) folder, run the [visu_2D.jl](scripts_start/visu_2D.jl) script which should produce a heatmap of a Gaussian distribution in 2D.

11. Finally, you should at this stage be able to run the following scripts to make sure MPI-based GPU selection and GPU-aware MPI is running as expected in Julia. Exit Julia and go to the [scripts_start](scripts_start) folder:
    ```
    cd scripts_start
    ```
    Run the [hello_mpi_gpu.jl](scripts_start/hello_mpi_gpu.jl) script to make sure GPU selection works as expected:
    ```
    mpirun -np 4 -mca btl_openib_warn_default_gid_prefix 0 julia --project hello_mpi_gpu.jl
    ```
    Run the [alltoall_mpi_gpu.jl](scripts_start/alltoall_mpi_gpu.jl) script to verify GPU-aware MPI is working:
    ```
    mpirun -np 4 -mca btl_openib_warn_default_gid_prefix 0 julia --project alltoall_mpi_gpu.jl
    ```


If you made it here you should be all set :rocket:

#### The small print
Note that the following config is already set in your `.bashrc` to prepare the correct environment:
```sh
# User specific aliases and functions
# load modules
module load julia cuda/11.4 openmpi/gcc83-314-c112
# Julia setup
alias juliap='julia --project'
# new Preferences.jl based config
export JULIA_LOAD_PATH="$JULIA_LOAD_PATH:/soft/julia/julia_prefs/"
export JULIA_CUDA_MEMORY_POOL=none
```
<br>
</details>

### Useful resources
- PDE on GPUs ETH Zurich course
- Julia Discourse (Julia Q&A)
- Julia Slack (Julia dev chat)

### Julia and HPC
Some words on the Julia at scale effort, the Julia HPC packages, and why Julia for HPC in general (two language barrier)
## Slot 2
**Hands-on I**

### Solving the transient 2D diffusion on the CPU I

### Solving the transient 2D diffusion on the CPU II

### Solving the transient 2D diffusion on GPU

### Channel flow in 2D
## Slot 3
**Hands-on II**
### Multi-CPU diffusion solver

### Multi-GPU diffusion solver

### Multi-GPU channel flow
## Slot 4
**OPTION 1**
### AD tools in Julia

### JVP calculations

### Advanced
#### Towards sensitivity kernels and adjoint solutions

#### The accelerated pseudo-transient method
