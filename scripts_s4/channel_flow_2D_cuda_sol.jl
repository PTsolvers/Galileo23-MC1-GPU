using Plots, Printf
using Plots.PlotMeasures
using CUDA

using Enzyme

@views av(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
macro avy(A)  esc(:(0.5 * ($A[iy, iz] + $A[iy+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[iy, iz] + $A[iy, iz+1]))) end
macro av4(A)  esc(:(0.25 * ($A[iy, iz] + $A[iy, iz+1] + $A[iy+1, iz] + $A[iy+1, iz+1]))) end
macro d_ya(A) esc(:($A[iy+1, iz] - $A[iy, iz])) end
macro d_za(A) esc(:($A[iy, iz+1] - $A[iy, iz])) end
macro d_yi(A) esc(:($A[iy+1, iz+1] - $A[iy, iz+1])) end
macro d_zi(A) esc(:($A[iy+1, iz+1] - $A[iy+1, iz])) end
macro eII()   esc(:(sqrt((0.5 * ((vx[iy+1, iz+1] - vx[iy, iz+1]) / dy + (vx[iy+1, iz] - vx[iy, iz]) / dy))^2 + (0.5 * ((vx[iy+1, iz+1] - vx[iy+1, iz]) / dz + (vx[iy, iz+1] - vx[iy, iz]) / dz))^2))) end

function update_ηeff!(ηeff, vx, k0, npow, ηreg, ηrel, dy, dz)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (iy <= size(ηeff, 1) && iz <= size(ηeff, 2)) ηeff[iy, iz] = ηeff[iy, iz] * (1.0 - ηrel) + ηrel / (1.0 ./ (k0 * @eII()^(npow - 1.0)) + 1.0 / ηreg) end
    return
end

function assign_ηeff!(ηeff, vx, k0, npow, ηreg, dy, dz)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (iy <= size(ηeff, 1) && iz <= size(ηeff, 2)) ηeff[iy, iz] = 1.0 / (1.0 ./ (k0 * @eII()^(npow - 1.0)) + 1.0 / ηreg) end
    return
end

function grad_assign_ηeff!(η̄eff,v̄x,ηeff, vx, k0, npow, ηreg, dy, dz)
    Enzyme.autodiff_deferred(Reverse,assign_ηeff!,Duplicated(ηeff,η̄eff),Duplicated(vx,v̄x),Const(k0),Const(npow),Const(ηreg),Const(dy),Const(dz))
    return
end

function update_τ!(ηeff, τxy, τxz, vx, dy, dz)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (iy <= size(τxy, 1) && iz <= size(τxy, 2)) τxy[iy, iz] = @avz(ηeff) * @d_yi(vx) / dy end
    if (iy <= size(τxz, 1) && iz <= size(τxz, 2)) τxz[iy, iz] = @avy(ηeff) * @d_zi(vx) / dz end
    return
end

function grad_update_τ!(τ̄xy, τ̄xz, η̄eff, v̄x, ηeff, τxy, τxz, vx, dy, dz)
    Enzyme.autodiff_deferred(Reverse,update_τ!,Duplicated(ηeff,η̄eff),Duplicated(τxy,τ̄xy),Duplicated(τxz,τ̄xz),Duplicated(vx,v̄x),Const(dy),Const(dz))
    return
end

function update_v!(vx, τxy, τxz, ηeff, ρg, sinα, dτ, dy, dz)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (iy <= size(vx, 1) - 2 && iz <= size(vx, 2) - 2) vx[iy+1, iz+1] = vx[iy+1, iz+1] + (@d_ya(τxy) / dy + @d_za(τxz) / dz + ρg * sinα) * dτ / @av4(ηeff) end
    return
end

function apply_bc!(vx)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (iy <= size(vx, 1) && iz == size(vx, 2)) vx[iy, iz] = vx[iy, iz-1] end
    if (iy == 1           && iz <= size(vx, 2)) vx[iy, iz] = vx[iy+1, iz] end
    return
end

function residual!(resv, τxy, τxz, ρg, sinα, dy, dz)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (iy <= size(resv, 1) && iz <= size(resv, 2)) resv[iy, iz] = @d_ya(τxy) / dy + @d_za(τxz) / dz + ρg * sinα end
    return
end

function grad_residual!(r̄, τ̄xy, τ̄xz, r, τxy, τxz, ρg, sinα, dy, dz)
    Enzyme.autodiff_deferred(Reverse,residual!,Duplicated(r,r̄),Duplicated(τxy,τ̄xy),Duplicated(τxz,τ̄xz),Const(ρg),Const(sinα),Const(dy),Const(dz))
    return
end

@views function main()
    # physics
    # non-dimensional
    npow     = 1.0 / 3.0
    sinα     = sin(π / 12)
    # dimensionally independent
    ly, lz   = 1.0, 1.0 # [m]
    k0       = 1.0      # [Pa*s^npow]
    ρg       = 1.0      # [Pa/m]
    # scales
    psc      = ρg * lz
    ηsc      = psc * (k0 / psc)^(1.0 / npow)
    # dimensionally dependent
    ηreg     = 1e4 * ηsc
    # numerics
    nz       = 64
    ny       = ceil(Int, nz * ly / lz)
    nthreads = (16, 16)
    nblocks  = cld.((ny, nz), nthreads)
    cfl      = 1 / 4.1
    ϵtol     = 1e-6
    ηrel     = 5e-1
    maxiter  = 20000max(ny, nz)
    ncheck   = 500max(ny, nz)
    # preprocessing
    dy, dz   = ly / ny, lz / nz
    yc, zc   = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    yv, zv   = av(yc), av(zc)
    dτ      = cfl * min(dy, dz)^2
    # init
    vx       = CUDA.zeros(Float64, ny, nz)
    ηeff     = CUDA.zeros(Float64, ny - 1, nz - 1)
    τxy      = CUDA.zeros(Float64, ny - 1, nz - 2)
    τxz      = CUDA.zeros(Float64, ny - 2, nz - 1)
    resv     = CUDA.zeros(Float64, ny - 2, nz - 2)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_ηeff!(ηeff, vx, k0, npow, ηreg, ηrel, dy, dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_τ!(ηeff, τxy, τxz, vx, dy, dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_v!(vx, τxy, τxz, ηeff, ρg, sinα, dτ, dy, dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks apply_bc!(vx)
        if iter % ncheck == 0
            CUDA.@sync @cuda threads=nthreads blocks=nblocks residual!(resv, τxy, τxz, ρg, sinα, dy, dz)
            err = maximum(abs.(resv)) * lz / psc
            push!(iters_evo, iter / nz); push!(errs_evo, err)
            p1 = heatmap(yc, zc, Array(vx)'; aspect_ratio=1, xlabel="y", ylabel="z", title="Vx", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, right_margin=10mm)
            p2 = heatmap(yv, zv, Array(ηeff)'; aspect_ratio=1, xlabel="y", ylabel="z", title="ηeff", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, colorbar_scale=:log10)
            p3 = plot(iters_evo, errs_evo; xlabel="niter/nx", ylabel="err", yscale=:log10, framestyle=:box, legend=false, markershape=:circle)
            display(plot(p1, p2, p3; size=(1200, 400), layout=(1, 3), bottom_margin=10mm, left_margin=10mm))
            @printf("  #iter/nz=%.1f, err=%1.3e\n", iter / nz, err)
        end
        iter += 1
    end
    return
end

main()