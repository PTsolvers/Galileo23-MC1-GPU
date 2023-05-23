using Enzyme
using CUDA

function residual_1!(R,C,dc,dx)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    ??
    return
end

function grad_residual_1!(R̄,C̄,R,C,dc,dx)
    Enzyme.autodiff_deferred(Reverse,residual_1!,Duplicated(R,R̄),Duplicated(C,C̄),Const(dc),Const(dx))
    return
end

function update_q!(qx,C,dc,dx)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    ??
    return
end

function grad_update_q!(q̄x,C̄,qx,C,dc,dx)
    ??
    return
end

function update_R!(R,qx,dx)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    ??
    return
end

function grad_update_R!(R̄,q̄x,R,qx,dx)
    ??
    return
end

function residual!(R,C,qx,dc,dx,nblocks,nthreads)
    ??
    return
end

function grad_residual!(R̄,q̄x,C̄,R,C,qx,dc,dx,nblocks,nthreads)
    ??
    return
end

function main()
    nx  = 10
    C   = CUDA.rand(Float64,nx)
    C̄   = CUDA.zeros(Float64,nx)
    C̄_1 = CUDA.zeros(Float64,nx)
    qx  = CUDA.zeros(Float64,nx-1)
    q̄x  = CUDA.zeros(Float64,nx-1)
    R   = CUDA.zeros(Float64,nx)
    R̄   = CUDA.ones(Float64,nx)
    R̄_1 = CUDA.ones(Float64,nx)
    dx  = 1.0/nx
    dc  = 1.0

    nthreads = 256
    nblocks  = cld(nx,nthreads)

    CUDA.@sync @cuda threads=nthreads blocks=nblocks residual_1!(R,C,dc,dx)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks grad_residual_1!(R̄_1,C̄_1,R,C,dc,dx)

    CUDA.@sync @cuda threads=nthreads blocks=nblocks update_q!(qx,C,dc,dx)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks update_R!(R,qx,dx)

    residual!(??)
    grad_residual!(??)

    @assert C̄ ≈ C̄_1

    return
end

main()

