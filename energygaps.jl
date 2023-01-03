using QuantumOptics
using Distributions
using LinearAlgebra
using Combinatorics


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

function degenerateEnergyGaps(H::AbstractOperator)
    #routine checks energy gaps non degeneracy condition

    #find eigenvalues, rounding to avoid errors with imaginary part
    energies = round.(eigenenergies(dense(H)); digits = 12)
    println(energies)
    #find all the possible energy gaps 
    #note: gaps E1-E2 and E2-E1 are counted as a single one
    gapPairs = collect(combinations(energies, 2))

    #define energy gaps array and find energy gaps
    energyGaps = Array{Float64}(undef, length(gapPairs))
    for j in 1:length(gapPairs)
        energyGaps[j] = gapPairs[j][1]-gapPairs[j][2]
    end
    
    #count number of unique energy gaps
    numUnique = length(unique(energyGaps))

    #if number of unique elements matches number of energy gaps
    print("Number of eigengaps: ")
    println(length(energyGaps))

    print("Unique eigengaps: ")
    println(numUnique)
   
    if numUnique == length(energyGaps)
        println("Hamiltonian does *not* have degenerate eigengaps!")
        return false
    else 
        println("Hamiltonian has degenerate eigengaps!")
        return true
    end
end

function main()
    #set number of environmental spins
    envN = 2

    #define single spin basis, environment basis and composite basis 
    sb = SpinBasis(1//2)
    eb = SpinBasis(1//2)^envN
    seb = sb ⊗ eb

    #define parameters
    ωs = 1.
    ω = rand(Uniform(1.1, 2.), envN)
    γ = rand(Uniform(0.01, 0.05), envN)
    #ω = [1.14, 1.12, 1.13, 1.17]
    #γ = [0.028, 0.026, 0.021, 0.024]
    #define Hamiltonians
    HE = envHamiltonian(ω,sigmaz(sb), eb)
    HS = ωs*sigmaz(sb)
    HI = intHamiltonianExc(γ, eb)
    Htot = HS ⊗ one(eb) + one(sb) ⊗ HE + HI

    #check if Hamiltonian has degenerate energy gaps. 
    #routine returns true if H is degenerate (gaps).
    degenerateEnergyGaps(Htot)
end 

main()