using QuantumOptics
using Distributions
using LinearAlgebra
using Combinatorics
using StatsBase
using PyPlot

include("iepRoutines.jl")

function equilibriumState(envN::Int, param, ρs::AbstractOperator)

    #unpack parameters
    ω, ωs, γ, β = param
    #initialize Basis 
    sb, eb, seb = initializeSpinBasis(envN)

    #construct Hamiltonians
    HE = envHamiltonian(ω,sigmaz(sb), eb)
    HS = ωs*sigmaz(sb)
    HI = intHamiltonianExc(γ, eb)
    Htot = HS ⊗ one(eb) + one(sb) ⊗ HE + HI

    #define initial states
    ρe = gibbsThermalState(β,HE)
    ρ = ρs ⊗ ρe

    #find equilibrium state
    ρeq, bound, εmin, βeq = equilibriumLinden(Htot,ρ)
    boundt = equilibriumBoundShort(bound, εmin, 1000., envN)

    return ρeq, bound, εmin, βeq

end

function sampledEquilibriumState(envNmax, ρs, β, ωmin,ωmax,γmin,γmax,sampling = 10)
    βeq = zeros(envNmax-1)
    bounds = zeros(envNmax-1)
    envNarray = zeros(envNmax-1)

    #equilibrium state
    for envN in 2:envNmax
        for sample in 1:sampling
            #get new random parameters
            ω = rand(Uniform(ωmin,ωmax), envN)
            γ = rand(Uniform(γmin,γmax), envN)
            ρeq, bound, εmin, βe = equilibriumState(envN, [ω, 1.,  γ, β], ρs)
            βeq[envN-1] += βe
            bounds[envN-1] += bound
        end
        envNarray[envN-1] = envN
    end
    βeq = βeq/sampling
    bounds = bounds/sampling

    return βeq, bounds
end

function printArrayMathematica(array)
    print("{")
    for i in 1:length(array)
        if(i<length(array))
            print(array[i], ", ")
        else
            println(array[i], "}")
        end
    end
end
function main()

    #initial state
    sb = SpinBasis(1//2)
    #ρs = projector(spinup(sb)) 
    #ρs = normalize(projector(spinup(sb)) + projector(spindown(sb)))
    ρs = normalize(projector(normalize(spinup(sb) + spindown(sb))))
    #ρs = gibbsThermalState(0.75, sigmaz(sb))
    
    #parameters
    #it is interesting to consider different regimes of parameters, in particular we are interested in 
    #small and large detuning (frequency)
    #weak and strong coupling
    envNmax = 6
    ωs = 1.
    ωmin = 1.1
    ωmax = 1.2
    γmin = 0.1
    γmax = 0.5
    β = 0.5

    #equilibrium state 
    βeq,bounds =  sampledEquilibriumState(envNmax, ρs, β, ωmin, ωmax, γmin, γmax)
    println("Results for equilibrium state")
    println("Environment β: ", β)
    println("Initial state: thermalstate with β = 0.4")
    printArrayMathematica(collect(2:envNmax))
    printArrayMathematica(bounds)
    printArrayMathematica(βeq)

    βeq,bounds =  sampledEquilibriumState(envNmax, ρs, β, ωmin, ωmax, 0.01, 0.05)
    println("Results for equilibrium state")
    println("Environment β: ", β)
    println("Initial state: thermalstate with β = 0.4")
    printArrayMathematica(collect(2:envNmax))
    printArrayMathematica(bounds)
    printArrayMathematica(βeq)



end

main()