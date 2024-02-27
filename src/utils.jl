"""
    __prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
__prevfloat_tdir(x, x0, x1) = ifelse(x1 > x0, prevfloat(x), nextfloat(x))

"""
    __nextfloat_tdir(x, x0, x1)

Move `x` one floating point towards x1.
"""
__nextfloat_tdir(x, x0, x1) = ifelse(x1 > x0, nextfloat(x), prevfloat(x))

"""
    __max_tdir(a, b, x0, x1)

Return the maximum of `a` and `b` if `x1 > x0`, otherwise return the minimum.
"""
__max_tdir(a, b, x0, x1) = ifelse(x1 > x0, max(a, b), min(a, b))

function __value_and_jacobian! end

"""
    value_and_jacobian(ad, f, y, x, p, cache; J = nothing)

Compute `f(x), d/dx f(x)` in the most efficient way based on `ad`. None of the arguments
except `cache` (& `J` if not nothing) are mutated.
"""
function value_and_jacobian(ad, f::F, y, x, p, cache; J = nothing) where {F}
    if isinplace(f)
        if SciMLBase.has_jac(f)
            f.jac(J, x, p)
            f(y, x, p)
            return y, J
        end
        __value_and_jacobian!(Val(true), ad, J, @closure((du, u)->f(du, u, p)), y, x, cache)
    else
        SciMLBase.has_jac(f) && return f(x, p), f.jac(x, p)
        __value_and_jacobian!(Val(false), ad, J, Base.Fix2(f, p), y, x, cache)
    end
end

"""
    jacobian_cache(ad, f, y, x, p) --> J, cache

Returns a Jacobian Matrix and a cache for the Jacobian computation.
"""
function jacobian_cache(ad, f::F, y, x::X, p) where {F, X <: AbstractArray}
    if isinplace(f)
        if SciMLBase.has_jac(f)
            return (similar(y, promote_type(eltype(x), eltype(y)), length(y), length(x)),
                nothing)
        end
        return __jacobian_cache(Val(true), ad, @closure((du, u)->f(du, u, p)), y, x)
    else
        SciMLBase.has_jac(f) && return nothing, nothing
        return __jacobian_cache(Val(false), ad, Base.Fix2(f, p), y, x)
    end
end

jacobian_cache(ad, f::F, y, x::Number, p) where {F} = nothing, nothing

__jacobian_cache(::Val, ad, f::F, y, x) where {F} = __test_loaded_backend(ad, eltype(x))

function compute_jacobian_and_hessian(ad, prob, fx, x)
    __test_loaded_backend(ad, x)
    error("`compute_jacobian_and_hessian` not implemented for $(ad).")
end

__init_identity_jacobian(u::Number, fu, α = true) = oftype(u, α)
__init_identity_jacobian!!(J::Number) = one(J)
function __init_identity_jacobian(u, fu, α = true)
    J = similar(u, promote_type(eltype(u), eltype(fu)), length(fu), length(u))
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= eltype(J)(α)
    return J
end
function __init_identity_jacobian!!(J)
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= one(eltype(J))
    return J
end
function __init_identity_jacobian!!(J::AbstractVector)
    fill!(J, one(eltype(J)))
    return J
end
function __init_identity_jacobian(u::StaticArray, fu, α = true)
    S1, S2 = length(fu), length(u)
    J = SMatrix{S1, S2, eltype(u)}(I * α)
    return J
end
function __init_identity_jacobian!!(J::SMatrix{S1, S2}) where {S1, S2}
    return SMatrix{S1, S2, eltype(J)}(I)
end
function __init_identity_jacobian!!(J::SVector{S1}) where {S1}
    return ones(SVector{S1, eltype(J)})
end

@inline _vec(v) = vec(v)
@inline _vec(v::Number) = v
@inline _vec(v::AbstractVector) = v

@inline _restructure(y::Number, x::Number) = x
@inline _restructure(y, x) = ArrayInterface.restructure(y, x)

@inline function _get_fx(prob::NonlinearLeastSquaresProblem, x)
    isinplace(prob) && prob.f.resid_prototype === nothing &&
        error("Inplace NonlinearLeastSquaresProblem requires a `resid_prototype`")
    return _get_fx(prob.f, x, prob.p)
end
@inline _get_fx(prob::NonlinearProblem, x) = _get_fx(prob.f, x, prob.p)
@inline function _get_fx(f::NonlinearFunction, x, p)
    if isinplace(f)
        if f.resid_prototype !== nothing
            T = eltype(x)
            return T.(f.resid_prototype)
        else
            fx = similar(x)
            f(fx, x, p)
            return fx
        end
    else
        return f(x, p)
    end
end

# Termination Conditions Support
# Taken directly from NonlinearSolve.jl
# The default here is different from NonlinearSolve since the userbases are assumed to be
# different. NonlinearSolve is more for robust / cached solvers while SimpleNonlinearSolve
# is meant for low overhead solvers, users can opt into the other termination modes but the
# default is to use the least overhead version.
function init_termination_cache(abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(abstol, reltol, du, u, AbsNormTerminationMode())
end
function init_termination_cache(abstol, reltol, du, u, tc::AbstractNonlinearTerminationMode)
    tc_cache = init(du, u, tc; abstol, reltol)
    return (NonlinearSolveBase.get_abstol(tc_cache),
        NonlinearSolveBase.get_reltol(tc_cache), tc_cache)
end

function check_termination(tc_cache, fx, x, xo, prob, alg)
    return check_termination(tc_cache, fx, x, xo, prob, alg,
        NonlinearSolveBase.get_termination_mode(tc_cache))
end

function check_termination(tc_cache, fx, x, xo, prob, alg,
        mode::AbstractNonlinearTerminationMode)
    if Bool(tc_cache(fx, x, xo))
        if mode isa AbstractSafeBestNonlinearTerminationMode
            if isinplace(prob)
                prob.f(fx, x, prob.p)
            else
                fx = prob.f(x, prob.p)
            end
        end
        return build_solution(prob, alg, x, fx; retcode = tc_cache.retcode)
    end
    return nothing
end

@inline value(x) = x

@inline __eval_f(prob, fx, x) = isinplace(prob) ? (prob.f(fx, x, prob.p); fx) :
                                prob.f(x, prob.p)

# Unalias
@inline __maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
@inline function __maybe_unaliased(x::AbstractArray, alias::Bool)
    # Spend time coping iff we will mutate the array
    (alias || !ArrayInterface.can_setindex(typeof(x))) && return x
    return deepcopy(x)
end

# Decide which AD backend to use
@inline function __get_concrete_autodiff(prob, ad::ADTypes.AbstractADType; kwargs...)
    return __test_loaded_backend(ad, prob.u0)
end
@inline function __get_concrete_autodiff(prob, ::Nothing; polyester::Val{P} = Val(true),
        kwargs...) where {P}
    if P && __is_extension_loaded(Val(:PolyesterForwardDiff)) &&
       __can_dual(eltype(prob.u0)) && !(prob.u0 isa Number) &&
       ArrayInterface.can_setindex(prob.u0)
        return AutoPolyesterForwardDiff()
    elseif __is_extension_loaded(Val(:ForwardDiff)) && __can_dual(eltype(prob.u0))
        return AutoForwardDiff()
    elseif __is_extension_loaded(Val(:FiniteDiff))
        return AutoFiniteDiff()
    else
        error("No AD Package is Loaded: Please install and load `PolyesterForwardDiff.jl`, \
               `ForwardDiff.jl`, or `FiniteDiff.jl`.")
    end
end

for backend in (:PolyesterForwardDiff, :ForwardDiff, :FiniteDiff, :Zygote)
    adtype = Symbol(:Auto, backend)
    msg1 = "ADType: `$(adtype)` is not compatible with !(ForwardDiff.can_dual(eltype(x)))."
    msg2 = "ADType: `$(adtype)` requires the `$(backend).jl` package to be loaded."
    @eval begin
        function __test_loaded_backend(ad::$(adtype), x)
            if __is_extension_loaded($(Val(backend)))
                __compatible_ad_with_eltype(ad, x) && return ad
                error($(msg1))
            end
            error($(msg2))
        end
    end
end

function __can_dual end
@inline __compatible_ad_with_eltype(::Union{AutoForwardDiff, AutoPolyesterForwardDiff}, x) = __can_dual(eltype(x))
@inline __compatible_ad_with_eltype(::ADTypes.AbstractADType, x) = true

@inline __reshape(x::Number, args...) = x
@inline __reshape(x::AbstractArray, args...) = reshape(x, args...)

# Extension
function __zygote_compute_nlls_vjp end
