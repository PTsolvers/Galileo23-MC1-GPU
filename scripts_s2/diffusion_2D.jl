using Plots, Printf
using Plots.PlotMeasures

@views av(A)  = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avy(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avz(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views av4(A) = 0.25 .* (A[1:end-1, 1:end-1] .+ A[1:end-1, 2:end] .+ A[2:end, 1:end-1] .+ A[2:end, 2:end])

@views function main()
    # physics
    ly, lz  = 1.0, 1.0
    d0      = 1.0
    # numerics
    nz      = 64
    ny      = ceil(Int, nz * ly / lz)
    cfl     = 1 / ??
    maxiter = 200
    ncheck  = 20
    # preprocessing
    dy, dz  = ly / ny, lz / nz
    yc, zc  = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    dτ      = cfl * min(dy, dz)^2
    # init
    C       = @. exp(-yc^2 / 0.02 - (zc' - lz / 2)^2 / 0.02)
    D       = d0 .* ones(??)
    qy      = zeros(??)
    qz      = zeros(??)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; iter = 1
    while iter <= maxiter
        qy .= ??
        qz .= ??
        C ??
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