module SimpleNonlinearSolveDiffResultsForwardDiffExt

import PrecompileTools: @compile_workload, @setup_workload

import ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
import ArrayInterface, SciMLBase, SimpleNonlinearSolve, DiffResults, ForwardDiff
import FastClosures: @closure
import LinearAlgebra: mul!
import SciMLBase: IntervalNonlinearProblem, NonlinearProblem, NonlinearLeastSquaresProblem,
                  solve
import SimpleNonlinearSolve: AbstractSimpleNonlinearSolveAlgorithm, __nlsolve_ad,
                             __nlsolve_dual_soln, __nlsolve_∂f_∂p, __nlsolve_∂f_∂u,
                             Bisection, Brent, Alefeld, Falsi, ITP, Ridder
import StaticArraysCore: StaticArray, SArray, Size

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:ForwardDiff}) = true

@inline SimpleNonlinearSolve.__can_dual(x) = ForwardDiff.can_dual(x)

@inline SimpleNonlinearSolve.value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
@inline SimpleNonlinearSolve.value(x::AbstractArray{<:ForwardDiff.Dual}) = map(
    ForwardDiff.value, x)

include("jacobian.jl")
include("hessian.jl")
include("forward_ad.jl")

@setup_workload begin
    for T in (Float32, Float64)
        prob_no_brack_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_no_brack_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))
        prob_no_brack_oop = NonlinearProblem{false}((u, p) -> u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))

        algs = [SimpleNonlinearSolve.SimpleNewtonRaphson(; autodiff = AutoForwardDiff()),
            SimpleNonlinearSolve.SimpleTrustRegion(; autodiff = AutoForwardDiff())]

        algs_no_iip = [SimpleNonlinearSolve.SimpleHalley(; autodiff = AutoForwardDiff())]

        @compile_workload begin
            for alg in algs
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_iip, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end

            for alg in algs_no_iip
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end
        end
    end
end

end
