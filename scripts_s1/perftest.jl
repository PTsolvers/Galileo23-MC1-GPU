using CUDA, BenchmarkTools

macro d2_xi(A) esc(:(($A[ix+2, iy+1] - $A[ix+1, iy+1]) - ($A[ix+1, iy+1] - $A[ix, iy+1]))) end
macro d2_yi(A) esc(:(($A[ix+1, iy+2] - $A[ix+1, iy+1]) - ($A[ix+1, iy+1] - $A[ix+1, iy]))) end
macro inn(A)  esc(:($A[ix+1, iy+1])) end

function diffusion_step!(C2, C, D, dt, dx, dy)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (ix <= size(C, 1) - 2 && iy <= size(C, 2) - 2)
        @inn(C2) = @inn(C) + dt * @inn(D) * (@d2_xi(C) / dx / dx + @d2_yi(C) / dy / dy)
    end
    return
end

function perftest()
    nx = ny = 512 * 64
    C  = CUDA.rand(Float64, nx, ny)
    D  = CUDA.rand(Float64, nx, ny)
    dx = dy = dt = rand()
    C2 = copy(C)
    nthreads = (16, 16)
    nblocks  = cld.((nx, ny), nthreads)
    t_it = @belapsed begin
        CUDA.@sync @cuda threads=$nthreads blocks=$nblocks diffusion_step!($C2, $C, $D, $dt, $dx, $dy)
    end
    T_eff = (2 * 1 + 1) * 1 / 1e9 * nx * ny * sizeof(Float64) / t_it
    println("T_eff = $(T_eff) GiB/s using CUDA.jl on a Nvidia A100 GPU")
    println("So that's cool. We are getting close to hardware limit, running at $(T_eff/1355*100) % of memory copy! 🚀")
    return
end

perftest()