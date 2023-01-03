using QuantumOptics
using Distributions
using LinearAlgebra
using Combinatorics
using StatsBase
using PyPlot

#### States and Hamiltonians
function gibbsThermalState(β, H)
    state = normalize(exp(-β*dense(H)))
    return state
end

function envHamiltonian(k::Vector{Float64}, operator::AbstractOperator, base::Basis)
    H = embed(base, 1, k[1]*operator)
    for j = 2:length(base.bases)
        H += embed(base, j, k[j]*operator)
    end
    return H
end

function intHamiltonianExc(k::Vector{Float64}, base::Basis)
    σp = sigmap(SpinBasis(1//2))
    σm = sigmam(SpinBasis(1//2))
    H = σp ⊗ embed(base, 1, k[1]* σm) + σm ⊗ embed(base, 1, k[1]* σp)
    for j = 2:length(base.bases)
        H += σp ⊗ embed(base, j, k[j]* σm) + σm ⊗ embed(base, j, k[j]* σp)
    end
    return H
end

#### Evolution operators
function unitaryEvolutionOp(H, t)
    Ut = exp(-1im*dense(H)*t)
    return Ut
end

function timedependentPropagator(Htot, tmin, tmax, dt)
    
    tspan = Vector(tmin:dt:tmax)
    U0 = identityoperator(Htot)
    function f(t, psi)
        α = 0.
        Ht = Htot + α*identityoperator(Htot.basis_l)*t
        return Ht
    end

    tout, Ut = timeevolution.schroedinger_dynamic(tspan, U0, f) 

    return Ut
end

function heisenbergEvolution(H, ρ0, tmin, tmax, dt)
    tspan = Vector(tmin:dt:tmax)
    tout, ρt = timeevolution.master(tspan,ρ0,H,[identityoperator(H)])
    ρst = Vector{AbstractOperator}(undef, length(ρt))
    for j in 1:length(ρt)
        ρst[j] = ptrace(ρt[j], [2:1:length(H.basis_l.shape);])
    end
    return tout, ρst
end


### Equilibrium State
function epsilonMin(countDict)
    nonDegenerateGaps = collect(keys(filter(p-> p.second == 1, countDict)))
    εmin = 100
    for j=1:length(nonDegenerateGaps), k=j+1:length(nonDegenerateGaps)
        ε = abs(nonDegenerateGaps[j] - nonDegenerateGaps[k])
        if ε < εmin
            εmin = ε
        else
            εmin = εmin
        end
    end
    return εmin
end

function degenerateEnergyGaps(energies::Vector{Float64}, precision=6)
   
    gapPairs = collect(combinations(energies, 2))
    energyGaps = Array{Float64}(undef, length(gapPairs))
    for j in 1:length(gapPairs)
        energyGaps[j] = gapPairs[j][1]-gapPairs[j][2]
    end
    energyGaps = round.(energyGaps; digits=precision)
    countMap = countmap(energyGaps)
    maxDegeneracy = last(sort(collect(countMap), by = x -> x[2]))[2]
    hasDegenerateGaps = !allunique(energyGaps)
    εmin = epsilonMin(countMap)
    return hasDegenerateGaps, maxDegeneracy, εmin
    
end

function equilibriumLinden(H::AbstractOperator, ρ::AbstractOperator)
    #1. check if Hamiltonian has degenerate energy gaps, 
    #   if not run equilibrium state routine
    energies, ek = eigenstates(dense(H))
    hasDegenerateGaps, maxDegeneracy, εmin = degenerateEnergyGaps(energies)
        
    #2.find ck2 coefficients
    ck2 = Array{Float64}(undef, length(energies))
    for j in 1:length(energies)
        ck2[j] = expect(ρ, ek[j])
    end
    #3.constuct omega state
    Ω = ck2[1]*projector(ek[1])
    for j in 2:length(ck2)
        Ω += ck2[j]*projector(ek[j])
    end
    #5.partial trace over environment
    ρeq = ptrace(Ω,[2:1:length(Ω.basis_l.shape);])

    #bounds
    deff = 1/sum(ck2.^2)
    ds = 2
    bound = 0.5*sqrt(ds*ds*maxDegeneracy*sum(ck2.^2))
    return ρeq, bound, εmin

end

function equilibriumBoundShort(bound, εmin, t, envN)
    tbound = bound*sqrt(1+ 8*(envN+1)/(εmin*t))
    return tbound
end

#### Metrics and Physicality
function isUnitary(op::AbstractOperator)
    if op*dagger(op) == identity(basis(op))
        return true
    else
        return false
    end
end


function main()
    #set number of environmental spins
    envN = 4

    #define single spin basis and 
    sb = SpinBasis(1//2)
    eb = SpinBasis(1//2)^envN
    seb = sb ⊗ eb

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
    ρs = projector(spinup(sb)) #system prepared in excited state
    ρe = gibbsThermalState(β,HE)
    ρ = ρs ⊗ ρe

    #equilibrium state
    ρeq, bound, εmin = equilibriumLinden(Htot,ρ)
    boundt = equilibriumBoundShort(bound, εmin, 500., envN)

    println(bound, " ", boundt)

    #Heisenberg evolution drho = -i[rho, H]
    tout, ρstmaster = heisenbergEvolution(Htot, ρ, 0, 500, 2)

    #Trace distance evolution
    D = Array{Float64}(undef, length(ρstmaster))
    for j in 1:length(ρstmaster)
        D[j] = tracedistance(ρstmaster[j],ρeq)
    end

    #plot trace distance using matplotlib
    plot(tout, D)
    show()
    

    
    #define time-dependent evolution operator
    #using a trivial time-dependence to check if the solver works as inteded
    # timedependentPropagator(independentHamiltonian, tmin, tmax, dt)
    #Ut = timedependentPropagator(Htot, 0, 10, 0.1)

    #evolved state
    #ρt = Ut*ρ*dagger(Ut)
    #ρst = ptrace(ρt, [2,3])

end 

main()