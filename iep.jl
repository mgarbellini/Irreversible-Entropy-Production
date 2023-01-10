using QuantumOptics
using Distributions
using LinearAlgebra
using Combinatorics
using StatsBase
using PyPlot

include("iepRoutines.jl")

function main()
    #set number of environmental spins
    envN = 4

    #define single spin basis and composite system basis
    sb,eb,seb = initializeSpinBasis(envN)

    #define parameters
    ωs = 1.
    ω = rand(Uniform(1.1, 1.3), envN)
    γ = rand(Uniform(0.01, 0.05), envN)
    β = 0.5
    t = 10.0
    tmax = 100.

    #define Hamiltonians
    HE = envHamiltonian(ω,sigmaz(sb), eb)
    HS = ωs*sigmaz(sb)
    HI = intHamiltonianExc(γ, eb)
    Htot = HS ⊗ one(eb) + one(sb) ⊗ HE + HI


    #define initial states
    ρs = projector(spindown(sb)) #system prepared in excited state
    ρe = gibbsThermalState(β,HE)
    ρ = ρs ⊗ ρe
 

    #equilibrium state
    ρeq, bound, εmin, β = equilibriumLinden(Htot,ρ)
    boundt = equilibriumBoundShort(bound, εmin, 1000., envN)


    #println(bound, " ", boundt)

    #Heisenberg evolution drho = -i[rho, H]
    #tout, ρstmaster = heisenbergEvolution(Htot, ρ, 0, 500, 2)

    
    

    
    #define time-dependent evolution operator
    #using a trivial time-dependence to check if the solver works as inteded
    # timedependentPropagator(independentHamiltonian, tmin, tmax, dt)
    #Ut = timedependentPropagator(Htot, 0, 10, 0.1)

    #evolved state
    #ρt = Ut*ρ*dagger(Ut)
    #ρst = ptrace(ρt, [2,3])

end 

main()