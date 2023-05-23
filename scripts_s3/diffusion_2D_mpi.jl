# mpirun -np 4 -mca btl_openib_warn_default_gid_prefix 0 julia --project diffusion_2D_mpi.jl
using Plots, Printf
using Plots.PlotMeasures
using MPI

@views av(A)  = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avy(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avz(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views av4(A) = 0.25 .* (A[1:end-1, 1:end-1] .+ A[1:end-1, 2:end] .+ A[2:end, 1:end-1] .+ A[2:end, 2:end])

function cooperative_mpi_wait(req)
    done, status = false, MPI.STATUS_ZERO
    while !done
        done, status = MPI.Test(req, MPI.Status)
        yield()
    end
    return status
end

function cooperative_wait(task::Task, comm)
    while !Base.istaskdone(task)
        MPI.Iprobe(comm)
        yield()
    end
    wait(task)
end

function exchange_halo!(C, neighbors, send_bufs, recv_bufs, comm)
    for dim in 1:ndims(C)
        recv, send = Task[], Task[]
        for (side, rank) in enumerate(neighbors[dim])
            if rank == MPI.PROC_NULL continue end
            r = Threads.@spawn begin
                ihalo = side == 1 ? 1 : size(C, dim)
                halo = selectdim(C, dim, ihalo)
                recv_req = MPI.Irecv!(recv_bufs[dim][side], comm; source=rank)
                cooperative_mpi_wait(recv_req)
                copyto!(halo, recv_bufs[dim][side])
            end
            s = Threads.@spawn begin
                iborder = side == 1 ? 2 : size(C, dim) - 1
                border = selectdim(C, dim, iborder)
                copyto!(send_bufs[dim][side], border)
                send_req = MPI.Isend(send_bufs[dim][side], comm; dest=rank)
                cooperative_mpi_wait(send_req)
            end
            push!(recv, r)
            push!(send, s)
        end
        for (r, s) in zip(recv, send)
            cooperative_wait(r, comm)
            cooperative_wait(s, comm)
        end
    end
    return
end

function gather!(dst, src, comm; root=0)
    dims, _, _ = MPI.Cart_get(comm)
    dims = Tuple(dims)
    if MPI.Comm_rank(comm) == root
        # make subtype for gather
        subtype = MPI.Types.create_subarray(size(dst), size(src), (0, 0), MPI.Datatype(eltype(dst)))
        subtype = MPI.Types.create_resized(subtype, 0, size(src, 1) * Base.elsize(dst))
        MPI.Types.commit!(subtype)
        # make VBuffer for collective communication
        counts = fill(Cint(1), dims)
        displs = similar(counts)
        d = zero(Cint)
        for j in 1:dims[2]
            for i in 1:dims[1]
                displs[i, j] = d
                d += 1
            end
            d += (size(src, 2) - 1) * dims[2]
        end
        # transpose displs as cartesian communicator is row-major
        recvbuf = MPI.VBuffer(dst, vec(counts), vec(displs'), subtype)
        MPI.Gatherv!(src, recvbuf, comm; root)
    else
        MPI.Gatherv!(src, nothing, comm; root)
    end
    return
end

@views function main(; dims=(1, 1))
    MPI.Init()
    # create MPI communicator
    comm = MPI.Cart_create(MPI.COMM_WORLD, dims)
    me = MPI.Comm_rank(comm)
    neighbors = ntuple(Val(length(dims))) do idim
        MPI.Cart_shift(comm, idim - 1, 1)
    end
    coords = Tuple(MPI.Cart_coords(comm))
    # physics
    ly, lz  = 1.0, 1.0
    d0      = 1.0
    # numerics
    nz      = 64
    ny      = ceil(Int, nz * ly / lz)
    cfl     = 1 / 4.1
    maxiter = 500
    ncheck  = 20
    # preprocessing
    ny_g, nz_g = (ny - 2) * dims[1] + 2, (nz - 2) * dims[2] + 2
    dy, dz  = ly / ny_g, lz / nz_g
    dτ      = cfl * min(dy, dz)^2
    # init
    y0, z0  = coords[1] * (ny - 2) * dy, coords[2] * (nz - 2) * dz
    yc      = [y0 + iy * dy + dy / 2 - ly / 2 for iy = 1:ny]
    zc      = [z0 + iz * dz + dz / 2          for iz = 1:nz]
    C       = @. exp(-yc^2 / 0.02 - (zc' - lz / 2)^2 / 0.02)
    D       = d0 .* ones(ny - 1,nz - 1)
    qy      = zeros(ny - 1, nz - 2)
    qz      = zeros(ny - 2, nz - 1)
    # MPI buffers
    send_bufs = [[zeros(nz) for side in 1:2], [zeros(ny) for side in 1:2]]
    recv_bufs = deepcopy(send_bufs)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; iter = 1
    while iter <= maxiter
        qy .= .-avz(D) .* diff(C[:, 2:end-1], dims=1) ./ dy
        qz .= .-avy(D) .* diff(C[2:end-1, :], dims=2) ./ dz
        C[2:end-1, 2:end-1] .-= (diff(qy, dims=1) ./ dy .+ diff(qz, dims=2) ./ dz) .* dτ ./ av4(D)
        exchange_halo!(C, neighbors, send_bufs, recv_bufs, comm)
        if me == 0 && iter % ncheck == 0
            err = maximum(C)
            push!(iters_evo, iter / nz_g); push!(errs_evo, err)
            @printf("  #iter=%.1f, max(C)=%1.3e\n", iter, err)
        end
        iter += 1
    end

    C_g = me == 0 ? zeros(dims[1] * (ny - 2), dims[2] * (nz - 2)) : nothing

    MPI.Barrier(comm)
    gather!(C_g, C[2:end-1, 2:end-1], comm)

    if me == 0
        yg, zg = LinRange(-ly / 2 + dy / 2 + dy, ly / 2 - dy / 2 - dy, ny_g - 2), LinRange(dz / 2 + dz, lz - dz / 2 - dz, nz_g - 2)
        p1 = heatmap(yg, zg, C_g'; aspect_ratio=1, xlabel="y", ylabel="z", title="C", xlims=(-ly / 2 + dy, ly / 2 - dy), ylims=(dz, lz - dz), c=:turbo, clims=(0, 1), right_margin=10mm)
        p2 = plot(iters_evo, errs_evo; xlabel="niter", ylabel="max(C)", yscale=:log10, framestyle=:box, legend=false, markershape=:circle)
        plot(p1, p2; size=(800, 400), layout=(1, 2), bottom_margin=10mm, left_margin=10mm)
        png("C.png")
    end

    MPI.Finalize()
    return
end

main(; dims=(2, 2))