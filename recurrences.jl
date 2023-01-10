using QuantumOptics
using Distributions
using LinearAlgebra
using Combinatorics
using StatsBase
using PyPlot

include("iepRoutines.jl")

function expectTimeSeries(opList, state)
    expectList = Array{Float64}(undef, length(opList))
    for j in 1:length(opList)
        expectList[j] = real(expect(opList[j], state))
    end
    return expectList
end

function main()
    # Parameters
    envN = 4   
    β = 0.1

    ωs = 1.
    ωmin = 1.05
    ωmax = 1.1
    γmin = 0.1
    γmax = 0.5
    ω = rand(Uniform(ωmin,ωmax), envN)
    γ = rand(Uniform(γmin,γmax), envN)

    tmin = 0
    tmax = 1500
    dt = 10

    # Basis
    sb, eb, seb = initializeSpinBasis(envN)

    # Hamiltonians
    HE = envHamiltonian(ω,sigmaz(sb), eb)
    HS = ωs*sigmaz(sb)
    HI = intHamiltonianExc(γ, eb)
    Htot = HS ⊗ one(eb) + one(sb) ⊗ HE + HI

    # Initial States
    ρs = projector(spinup(sb))
    ρe = gibbsThermalState(β,HE)
    ρ = ρs ⊗ ρe 

    # Heisenberg evolution drho = -i[rho, H]
    tout, ρst = heisenbergEvolution(Htot, ρ, tmin, tmax, dt)
    
    # Probability of being in the excited state at time t
    pt = expectTimeSeries(ρst, spinup(sb))

    # Plotting evolution of expectation value
    plot(tout, pt)
    show()

end

main()