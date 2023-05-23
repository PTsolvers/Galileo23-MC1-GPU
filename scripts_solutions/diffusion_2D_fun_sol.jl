using Plots, Printf
using Plots.PlotMeasures

@views av(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
macro avy(A)  esc(:(0.5 * ($A[iy, iz] + $A[iy+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[iy, iz] + $A[iy, iz+1]))) end
macro av4(A)  esc(:(0.25 * ($A[iy, iz] + $A[iy, iz+1] + $A[iy+1, iz] + $A[iy+1, iz+1]))) end
macro d_ya(A) esc(:($A[iy+1, iz] - $A[iy, iz])) end
macro d_za(A) esc(:($A[iy, iz+1] - $A[iy, iz])) end
macro d_yi(A) esc(:($A[iy+1, iz+1] - $A[iy, iz+1])) end
macro d_zi(A) esc(:($A[iy+1, iz+1] - $A[iy+1, iz])) end

function update_q!(qy, qz, C, D, dy, dz)
    Threads.@threads for iz = 1:size(C, 2)
        for iy = 1:size(C, 1)
            if (iy <= size(qy, 1) && iz <= size(qy, 2)) qy[iy, iz] = -@avz(D) * @d_yi(C) / dy end
            if (iy <= size(qz, 1) && iz <= size(qz, 2)) qz[iy, iz] = -@avy(D) * @d_zi(C) / dz end
        end
    end
    return
end

function update_C!(C, qy, qz, D, dτ, dy, dz)
    Threads.@threads for iz = 1:size(C, 2)
        for iy = 1:size(C, 1)
            if (iy <= size(C, 1) - 2 && iz <= size(C, 2) - 2) C[iy+1, iz+1] = C[iy+1, iz+1] - (@d_ya(qy) / dy + @d_za(qz) / dz) * dτ / @av4(D) end
        end
    end
    return
end

@views function main()
    # physics
    ly, lz  = 1.0, 1.0
    d0      = 1.0
    # numerics
    nz      = 128
    ny      = ceil(Int, nz * ly / lz)
    cfl     = 1 / 4.1
    maxiter = 200
    ncheck  = 20
    # preprocessing
    dy, dz  = ly / ny, lz / nz
    yc, zc  = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    dτ      = cfl * min(dy, dz)^2
    # init
    C       = @. exp(-yc^2 / 0.02 - (zc' - lz / 2)^2 / 0.02)
    D       = d0 .* ones(ny - 1, nz - 1)
    qy      = zeros(ny - 1, nz - 2)
    qz      = zeros(ny - 2, nz - 1)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; iter = 1
    while iter <= maxiter
        update_q!(qy, qz, C, D, dy, dz)
        update_C!(C, qy, qz, D, dτ, dy, dz)
        if iter % ncheck == 0
            err = maximum(C)
            push!(iters_evo, iter / nz); push!(errs_evo, err)
            p1 = heatmap(yc, zc, C'; aspect_ratio=1, xlabel="y", ylabel="z", title="C", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, clims=(0, 1), right_margin=10mm)
            p2 = plot(iters_evo, errs_evo; xlabel="niter", ylabel="max(C)", yscale=:log10, framestyle=:box, legend=false, markershape=:circle)
            display(plot(p1, p2; size=(800, 400), layout=(1, 2), bottom_margin=10mm, left_margin=10mm))
            @printf("  #iter=%.1f, max(C)=%1.3e\n", iter, err)
        end
        iter += 1
    end
    return
end

main()