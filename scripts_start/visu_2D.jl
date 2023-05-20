using Plots, Printf
using Plots.PlotMeasures

@views function main()
    # physics
    ly, lz  = 1.0, 1.0
    # numerics
    nz      = 64
    ny      = ceil(Int, nz * ly / lz)
    # preprocessing
    dy, dz  = ly / ny, lz / nz
    yc, zc  = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    # init
    C       = @. exp(-yc^2 / 0.02 - (zc' - lz / 2)^2 / 0.02)
    # action
    p1 = heatmap(yc, zc, C'; aspect_ratio=1, xlabel="y", ylabel="z", title="C", xlims=(-ly / 2, ly / 2), ylims=(0, lz), c=:turbo, clims=(0, 1), right_margin=10mm)
    display(plot(p1; size=(400, 400), layout=(1, 1), bottom_margin=10mm, left_margin=10mm))
    return
end

main()