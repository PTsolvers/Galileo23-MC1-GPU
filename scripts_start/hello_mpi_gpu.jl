# mpirun -np 4 -mca btl_openib_warn_default_gid_prefix 0 julia --project hello_mpi_gpu.jl
using MPI, CUDA
MPI.Init()
comm = MPI.COMM_WORLD
me = MPI.Comm_rank(comm)
# select device
comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
me_l = MPI.Comm_rank(comm_l)
GPU_ID = CUDA.device!(me_l)
sleep(0.1me)
println("Hello world, I am $(me) of $(MPI.Comm_size(comm)) using $(GPU_ID)")
MPI.Barrier(comm)
MPI.Finalize()