using Plots, Printf
using Plots.PlotMeasures
using CUDA

@views av(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
macro avy(A)  esc(:(0.5 * ($A[iy, iz] + $A[iy+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[iy, iz] + $A[iy, iz+1]))) end
macro av4(A)  esc(:(0.25 * ($A[iy, iz] + $A[iy, iz+1] + $A[iy+1, iz] + $A[iy+1, iz+1]))) end
macro d_ya(A) esc(:($A[iy+1, iz] - $A[iy, iz])) end
macro d_za(A) esc(:($A[iy, iz+1] - $A[iy, iz])) end
macro d_yi(A) esc(:($A[iy+1, iz+1] - $A[iy, iz+1])) end
macro d_zi(A) esc(:($A[iy+1, iz+1] - $A[iy+1, iz])) end

function update_q!(??)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ??
    ??
    return
end

function update_C!(??)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ??
    return
end

@views function main()
    # physics
    ly, lz   = 1.0, 1.0
    d0       = 1.0
    # numerics
    nz       = 128
    ny       = ceil(Int, nz * ly / lz)
    nthreads = (16, 16)
    nblocks  = cld.((ny, nz), nthreads)
    cfl      = 1 / 4.1
    maxiter  = 200
    ncheck   = 20
    # preprocessing
    dy, dz   = ly / ny, lz / nz
    yc, zc   = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    dτ       = cfl * min(dy, dz)^2
    # init
    C        = CuArray(@. exp(-yc^2 / 0.02 - (zc' - lz / 2)^2 / 0.02))
    D        = d0 .* CUDA.ones(Float64, ??)
    qy       = CUDA.zeros(??)
    qz       = CUDA.zeros(??)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; iter = 1
    while iter <= maxiter
        CUDA.@sync @cuda threads=nthreads blocks=nblocks ??
        CUDA.@sync @cuda threads=nthreads blocks=nblocks ??
        if iter % ncheck == 0
            err = maximum(C)
            push!(iters_evo, iter / nz); push!(errs_evo, err)
            p1 = heatmap(yc, zc, Array(C)'; aspect_ratio=1, xlabel="y", ylabel="z", title="C", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, clims=(0, 1), right_margin=10mm)
            p2 = plot(iters_evo, errs_evo; xlabel="niter", ylabel="max(C)", yscale=:log10, framestyle=:box, legend=false, markershape=:circle)
            display(plot(p1, p2; size=(800, 400), layout=(1, 2), bottom_margin=10mm, left_margin=10mm))
            @printf("  #iter=%.1f, max(C)=%1.3e\n", iter, err)
        end
        iter += 1
    end
    return
end

main()