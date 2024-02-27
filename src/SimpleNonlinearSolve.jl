module SimpleNonlinearSolve

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes, ArrayInterface, NonlinearSolveBase, Reexport, LinearAlgebra, SciMLBase

    import ConcreteStructs: @concrete
    import FastClosures: @closure
    import MaybeInplace: @bb, setindex_trait, CanSetindex, CannotSetindex
    import NonlinearSolveBase: AbstractNonlinearTerminationMode,
                               AbstractSafeNonlinearTerminationMode,
                               AbstractSafeBestNonlinearTerminationMode,
                               get_termination_mode, NONLINEARSOLVE_DEFAULT_NORM
    import SciMLBase: AbstractNonlinearAlgorithm, build_solution, isinplace, _unwrap_val
    import StaticArraysCore: StaticArray, SVector, SMatrix, SArray, MArray, MMatrix, Size
end

@reexport using ADTypes, SciMLBase  # TODO: Reexport NonlinearSolveBase after the situation with NonlinearSolve.jl is resolved

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end

@inline __is_extension_loaded(::Val) = false

include("utils.jl")
include("linesearch.jl")

## Nonlinear Solvers
include("nlsolve/raphson.jl")
include("nlsolve/broyden.jl")
include("nlsolve/lbroyden.jl")
include("nlsolve/klement.jl")
include("nlsolve/trustRegion.jl")
include("nlsolve/halley.jl")
include("nlsolve/dfsane.jl")

## Interval Nonlinear Solvers
include("bracketing/bisection.jl")
include("bracketing/falsi.jl")
include("bracketing/ridder.jl")
include("bracketing/brent.jl")
include("bracketing/alefeld.jl")
include("bracketing/itp.jl")

# AD: Defined in Extension
## DONT REMOVE THESE: They are used in NonlinearSolve.jl
function __nlsolve_ad end
function __nlsolve_∂f_∂p end
function __nlsolve_∂f_∂u end
function __nlsolve_dual_soln end

## Default algorithm

# Set the default bracketing method to ITP
SciMLBase.solve(prob::IntervalNonlinearProblem; kwargs...) = solve(prob, ITP(); kwargs...)
function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Nothing, args...; kwargs...)
    return solve(prob, ITP(), args...; prob.kwargs..., kwargs...)
end

# By Pass the highlevel checks for NonlinearProblem for Simple Algorithms
# Using eval to prevent ambiguity
for pType in (NonlinearProblem, NonlinearLeastSquaresProblem)
    @eval begin
        function SciMLBase.solve(
                prob::$(pType), alg::AbstractSimpleNonlinearSolveAlgorithm, args...;
                sensealg = nothing, u0 = nothing, p = nothing, kwargs...)
            if sensealg === nothing && haskey(prob.kwargs, :sensealg)
                sensealg = prob.kwargs[:sensealg]
            end
            new_u0 = u0 !== nothing ? u0 : prob.u0
            new_p = p !== nothing ? p : prob.p
            return __internal_solve_up(
                prob, sensealg, new_u0, u0 === nothing, new_p, p === nothing,
                alg, args...; prob.kwargs..., kwargs...)
        end

        function __internal_solve_up(_prob::$(pType), sensealg, u0, u0_changed, p,
                p_changed, alg::AbstractSimpleNonlinearSolveAlgorithm, args...; kwargs...)
            prob = u0_changed || p_changed ? remake(_prob; u0, p) : _prob
            return SciMLBase.__solve(prob, alg, args...; kwargs...)
        end
    end
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_no_brack_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_no_brack_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))
        prob_no_brack_oop = NonlinearProblem{false}((u, p) -> u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))

        algs = [SimpleBroyden(), SimpleKlement(), SimpleDFSane(),
            SimpleLimitedMemoryBroyden(; threshold = 2)]

        @compile_workload begin
            for alg in algs
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_iip, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end
        end

        prob_brack = IntervalNonlinearProblem{false}((u, p) -> u * u - p,
            T.((0.0, 2.0)), T(2))
        algs = [Bisection(), Falsi(), Ridder(), Brent(), Alefeld(), ITP()]
        @compile_workload begin
            for alg in algs
                solve(prob_brack, alg, abstol = T(1e-2))
            end
        end
    end
end

export SimpleBroyden, SimpleDFSane, SimpleGaussNewton, SimpleHalley, SimpleKlement,
       SimpleLimitedMemoryBroyden, SimpleNewtonRaphson, SimpleTrustRegion
export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end # module
