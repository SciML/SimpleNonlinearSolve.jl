module SimpleNonlinearSolveFiniteDiffExt

import ADTypes: AutoFiniteDiff
import SciMLBase, SimpleNonlinearSolve, FiniteDiff
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

end
