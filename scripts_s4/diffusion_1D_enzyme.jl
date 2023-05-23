using Enzyme

function residual_1!(R,C,dc,dx)
    for ix in 2:length(R)-1
        R[ix] = dc*(C[ix-1] - 2.0 * C[ix] + C[ix+1])/dx^2
    end
end

function grad_residual_1!(R̄,C̄,R,C,dc,dx)
    Enzyme.autodiff(Reverse,residual_1!,Duplicated(R,R̄),Duplicated(C,C̄),Const(dc),Const(dx))
    return
end

function update_q!(qx,C,dc,dx)
    for ix in eachindex(qx)
        qx[ix] = -dc*(C[ix+1]-C[ix])/dx
    end
    return
end

function update_R!(R,qx,dx)
    for ix in 2:length(R)-1
        R[ix] = -(qx[ix]-qx[ix-1])/dx
    end
    return
end

function residual!(R,C,qx,dc,dx)
    update_q!(qx,C,dc,dx)
    update_R!(R,qx,dx)
    return
end

function grad_residual!(??)
    Enzyme.autodiff(Reverse,??)
    Enzyme.autodiff(Reverse,??)
    return
end

function main()
    nx  = 10
    C   = rand(nx)
    C̄   = zeros(nx)
    C̄_1 = zeros(nx)
    qx  = zeros(nx-1)
    q̄x  = zeros(nx-1)
    R   = zeros(nx)
    R̄   = ones(nx)
    R̄_1 = ones(nx)
    dx  = 1.0/nx
    dc  = 1.0

    residual_1!(R,C,dc,dx)
    grad_residual_1!(R̄_1,C̄_1,R,C,dc,dx)

    update_q!(qx,C,dc,dx)
    update_R!(R,qx,dx)

    residual!(R,C,qx,dc,dx)
    grad_residual!(??)

    @assert C̄ ≈ C̄_1

    return
end

main()

