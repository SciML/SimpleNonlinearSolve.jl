module SimpleNonlinearSolveDiffResultsForwardDiffExt

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

end
