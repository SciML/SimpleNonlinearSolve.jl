module SimpleNonlinearSolveFiniteDiffExt

import PrecompileTools: @compile_workload, @setup_workload

import ADTypes: AutoFiniteDiff
import SciMLBase, SimpleNonlinearSolve, FiniteDiff
import SciMLBase: NonlinearProblem, NonlinearLeastSquaresProblem, solve
import StaticArraysCore: SArray

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:FiniteDiff}) = true

# Jacobian
function SimpleNonlinearSolve.__jacobian_cache(::Val{iip}, ad::AutoFiniteDiff, f::F, y,
        x) where {iip, F}
    cache = FiniteDiff.JacobianCache(copy(x), copy(y), copy(y), ad.fdtype)
    J = iip ? similar(y, promote_type(eltype(x), eltype(y)), length(y), length(x)) :
        nothing
    return J, cache
end
function SimpleNonlinearSolve.__jacobian_cache(::Val, ad::AutoFiniteDiff, f::F, y,
        x::SArray) where {F}
    return nothing, nothing
end

function SimpleNonlinearSolve.__value_and_jacobian!(
        ::Val{iip}, ad::AutoFiniteDiff, J, f::F, y, x, cache) where {iip, F}
    x isa Number && return (f(x), FiniteDiff.finite_difference_derivative(f, x, ad.fdtype))
    if iip
        FiniteDiff.finite_difference_jacobian!(J, f, x, cache)
        f(y, x)
        return y, J
    end
    cache === nothing && return f(x), FiniteDiff.finite_difference_jacobian(f, x)
    return f(x), FiniteDiff.finite_difference_jacobian(f, x, cache)
end

# Hessian
function SimpleNonlinearSolve.compute_jacobian_and_hessian(
        ad::AutoFiniteDiff, prob, _, x::Number)
    fx = prob.f(x, prob.p)
    J_fn = x -> FiniteDiff.finite_difference_derivative(Base.Fix2(prob.f, prob.p), x,
        ad.fdtype)
    dfx = J_fn(x)
    d2fx = FiniteDiff.finite_difference_derivative(J_fn, x, ad.fdtype)
    return fx, dfx, d2fx
end

function SimpleNonlinearSolve.compute_jacobian_and_hessian(
        ad::AutoFiniteDiff, prob, fx, x)
    if SciMLBase.isinplace(prob)
        error("Inplace version for Nested FiniteDiff Not Implemented Yet!")
    else
        f = Base.Fix2(prob.f, prob.p)
        fx = f(x)
        J_fn = x -> FiniteDiff.finite_difference_jacobian(f, x, ad.fdtype)
        dfx = J_fn(x)
        d2fx = FiniteDiff.finite_difference_jacobian(J_fn, x, ad.fdtype)
        return fx, dfx, d2fx
    end
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_no_brack_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_no_brack_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))
        prob_no_brack_oop = NonlinearProblem{false}((u, p) -> u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))

        algs = [SimpleNonlinearSolve.SimpleNewtonRaphson(; autodiff = AutoFiniteDiff()),
            SimpleNonlinearSolve.SimpleTrustRegion(; autodiff = AutoFiniteDiff())]

        algs_no_iip = [SimpleNonlinearSolve.SimpleHalley(; autodiff = AutoFiniteDiff())]

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
