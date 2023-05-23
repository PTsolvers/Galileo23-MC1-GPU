using Plots, Printf
using Plots.PlotMeasures

@views av(A)   = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avy(A)  = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avz(A)  = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views av4(A)  = 0.25 .* (A[1:end-1, 1:end-1] .+ A[1:end-1, 2:end] .+ A[2:end, 1:end-1] .+ A[2:end, 2:end])
@views bc2!(A) = (A[[1, end], :] .= A[[2, end - 1], :]; A[:, [1, end]] .= A[:, [2, end - 1]])

macro eII() esc(:(sqrt.((avz(diff(vx, dims=1) ./ dy)) .^ 2 .+ (avy(diff(vx, dims=2) ./ dz)) .^ 2))) end

@views function main()
    # physics
    # non-dimensional
    npow    = 1.0 / 3.0
    sinα    = sin(π / 12)
    # dimensionally independent
    ly, lz  = 1.0, 1.0 # [m]
    k0      = 1.0      # [Pa*s^npow]
    ρg      = 1.0      # [Pa/m]
    # scales
    psc     = ρg * lz
    ηsc     = psc * (k0 / psc)^(1.0 / npow)
    # dimensionally dependent
    ηreg    = 1e4 * ηsc
    # numerics
    nz      = 64
    ny      = ceil(Int, nz * ly / lz)
    cfl     = 1 / 4.1
    ϵtol    = 1e-6
    ηrel    = 5e-1
    maxiter = 20000max(ny, nz)
    ncheck  = 500max(ny, nz)
    # preprocessing
    dy, dz  = ly / ny, lz / nz
    yc, zc  = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    yv, zv  = av(yc), av(zc)
    dτ      = cfl * min(dy, dz)^2
    # init
    vx      = zeros(ny, nz)
    ηeff    = zeros(ny - 1, nz - 1)
    τxy     = zeros(ny - 1, nz - 2)
    τxz     = zeros(ny - 2, nz - 1)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        ηeff                  .= ηeff .* (1.0 - ηrel) .+ ηrel ./ (1.0 ./ (k0 .* @eII() .^ (npow - 1.0)) .+ 1.0 / ηreg)
        τxy                   .= avz(ηeff) .* diff(vx[:, 2:end-1], dims=1) ./ dy
        τxz                   .= avy(ηeff) .* diff(vx[2:end-1, :], dims=2) ./ dz
        vx[2:end-1, 2:end-1] .+= (diff(τxy, dims=1) ./ dy .+ diff(τxz, dims=2) ./ dz .+ ρg * sinα) .* dτ ./ av4(ηeff)
        vx[:, end]            .= vx[:, end-1]
        vx[1, :]              .= vx[2, :]
        if iter % ncheck == 0
            err = maximum(abs.(diff(τxy, dims=1) ./ dy .+ diff(τxz, dims=2) ./ dz .+ ρg * sinα)) * lz / psc
            push!(iters_evo, iter / nz); push!(errs_evo, err)
            p1 = heatmap(yc, zc, vx'; aspect_ratio=1, xlabel="y", ylabel="z", title="Vx", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, right_margin=10mm)
            p2 = heatmap(yv, zv, ηeff'; aspect_ratio=1, xlabel="y", ylabel="z", title="ηeff", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, colorbar_scale=:log10)
            p3 = plot(iters_evo, errs_evo; xlabel="niter/nx", ylabel="err", yscale=:log10, framestyle=:box, legend=false, markershape=:circle)
            display(plot(p1, p2, p3; size=(1200, 400), layout=(1, 3), bottom_margin=10mm, left_margin=10mm))
            @printf("  #iter/nz=%.1f, err=%1.3e\n", iter / nz, err)
        end
        iter += 1
    end
    return
end

main()