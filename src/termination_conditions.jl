abstract type AbstractNonlinearTerminationMode end
abstract type AbstractSafeNonlinearTerminationMode <: AbstractNonlinearTerminationMode end
abstract type AbstractSafeBestNonlinearTerminationMode <:
              AbstractSafeNonlinearTerminationMode end

"""
    SteadyStateDiffEqTerminationMode <: AbstractNonlinearTerminationMode

Check if all values of the derivative is close to zero wrt both relative and absolute
tolerance.

!!! danger

    This has been deprecated.
"""
struct SteadyStateDiffEqTerminationMode <: AbstractNonlinearTerminationMode
    function SteadyStateDiffEqTerminationMode()
        Base.depwarn("`SteadyStateDiffEqTerminationMode` is deprecated and isn't used \
                       in any upstream library. Remove uses of this.",
            :SteadyStateDiffEqTerminationMode)
        return new()
    end
end

"""
    SimpleNonlinearSolveTerminationMode <: AbstractNonlinearTerminationMode

Check if all values of the derivative is close to zero wrt both relative and absolute
tolerance. Or check that the value of the current and previous state is within the specified
tolerances.

!!! danger

    This has been deprecated.
"""
struct SimpleNonlinearSolveTerminationMode <: AbstractNonlinearTerminationMode
    function SimpleNonlinearSolveTerminationMode()
        Base.depwarn("`SimpleNonlinearSolveTerminationMode` is deprecated and isn't used \
                       in any upstream library. Remove uses of this.",
            :SimpleNonlinearSolveTerminationMode)
        return new()
    end
end

@inline set_termination_mode_internalnorm(mode, ::F) where {F} = mode

@inline __norm_type(::typeof(Base.Fix2(norm, Inf))) = :Inf
@inline __norm_type(::typeof(Base.Fix1(maximum, abs))) = :Inf
@inline __norm_type(::typeof(Base.Fix2(norm, 2))) = :L2
@inline __norm_type(::F) where {F} = F

const TERM_DOCS = Dict(
    :Norm => doc"``\| \Delta u \| \leq reltol \times \| \Delta u + u \|`` or ``\| \Delta u \| \leq abstol``.",
    :Rel => doc"``all \left(| \Delta u | \leq reltol \times | u | \right)``.",
    :RelNorm => doc"``\| \Delta u \| \leq reltol \times \| \Delta u + u \|``.",
    :Abs => doc"``all \left( | \Delta u | \leq abstol \right)``.",
    :AbsNorm => doc"``\| \Delta u \| \leq abstol``."
)

const __TERM_INTERNALNORM_DOCS = """
where `internalnorm` is the norm to use for the termination condition. Special handling is
done for `norm(_, 2)`, `norm(_, Inf)`, and `maximum(abs, _)`.

Default is left as `nothing`, which allows upstream frameworks to choose the correct norm
based on the problem type. If directly using the `init` API, a proper norm must be
provided"""

for name in (:Rel, :Abs)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        ``\\Delta u`` denotes the increment computed by the nonlinear solver and ``u`` denotes the solution.
        """
        struct $(struct_name) <: AbstractNonlinearTerminationMode end
    end
end

for name in (:Norm, :RelNorm, :AbsNorm)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        ``\\Delta u`` denotes the increment computed by the inner nonlinear solver.

        ## Constructor

            $($struct_name)(internalnorm = nothing)

        $($__TERM_INTERNALNORM_DOCS).
        """
        @concrete struct $(struct_name){F} <: AbstractNonlinearTerminationMode
            internalnorm

            $(struct_name)(f::F = nothing) where {F} = new{__norm_type(f), F}(f)
        end

        @inline function set_termination_mode_internalnorm(
                ::$(struct_name), internalnorm::F) where {F}
            return $(struct_name)(internalnorm)
        end
    end
end

for norm_type in (:Rel, :Abs), safety in (:Safe, :SafeBest)
    struct_name = Symbol(norm_type, safety, :TerminationMode)
    supertype_name = Symbol(:Abstract, safety, :NonlinearTerminationMode)

    doctring = safety == :Safe ?
               "Essentially [`$(norm_type)NormTerminationMode`](@ref) + terminate if there \
                has been no improvement for the last `patience_steps` + terminate if the \
                solution blows up (diverges)." :
               "Essentially [`$(norm_type)SafeTerminationMode`](@ref), but caches the best\
                solution found so far."

    @eval begin
        """
            $($struct_name) <: $($supertype_name)

        $($doctring)

        ## Constructor

            $($struct_name)(internalnorm = nothing; protective_threshold = nothing,
                patience_steps = 100, patience_objective_multiplier = 3,
                min_max_factor = 1.3, max_stalled_steps = nothing)

        $($__TERM_INTERNALNORM_DOCS).
        """
        @concrete struct $(struct_name){F, T <: Union{Nothing, Int}} <: $(supertype_name)
            internalnorm
            protective_threshold
            patience_steps::Int
            patience_objective_multiplier
            min_max_factor
            max_stalled_steps::T

            function $(struct_name)(f::F = nothing; protective_threshold = nothing,
                    patience_steps = 100, patience_objective_multiplier = 3,
                    min_max_factor = 1.3, max_stalled_steps = nothing) where {F}
                return new{__norm_type(f), typeof(max_stalled_steps), F,
                    typeof(protective_threshold), typeof(patience_objective_multiplier),
                    typeof(min_max_factor)}(f, protective_threshold, patience_steps,
                    patience_objective_multiplier, min_max_factor, max_stalled_steps)
            end
        end

        @inline function set_termination_mode_internalnorm(
                mode::$(struct_name), internalnorm::F) where {F}
            return $(struct_name)(internalnorm; mode.protective_threshold,
                mode.patience_steps, mode.patience_objective_multiplier,
                mode.min_max_factor, mode.max_stalled_steps)
        end
    end
end

@concrete mutable struct NonlinearTerminationModeCache{dep_retcode,
    M <: AbstractNonlinearTerminationMode,
    R <: Union{NonlinearSafeTerminationReturnCode.T, ReturnCode.T}}
    u
    retcode::R
    abstol
    reltol
    best_objective_value
    mode::M
    initial_objective
    objectives_trace
    nsteps::Int
    saved_values
    u0_norm
    step_norm_trace
    max_stalled_steps
    u_diff_cache
end

@inline get_termination_mode(cache::NonlinearTerminationModeCache) = cache.mode
@inline get_abstol(cache::NonlinearTerminationModeCache) = cache.abstol
@inline get_reltol(cache::NonlinearTerminationModeCache) = cache.reltol
@inline get_saved_values(cache::NonlinearTerminationModeCache) = cache.saved_values

function __update_u!!(cache::NonlinearTerminationModeCache, u)
    cache.u === nothing && return
    if cache.u isa AbstractArray && ArrayInterface.can_setindex(cache.u)
        copyto!(cache.u, u)
    else
        cache.u = u
    end
end

@inline __cvt_real(::Type{T}, ::Nothing) where {T} = nothing
@inline __cvt_real(::Type{T}, x) where {T} = real(T(x))

@inline _get_tolerance(η, ::Type{T}) where {T} = __cvt_real(T, η)
@inline function _get_tolerance(::Nothing, ::Type{T}) where {T}
    η = real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)
    return _get_tolerance(η, T)
end

function SciMLBase.init(du::Union{AbstractArray{T}, T}, u::Union{AbstractArray{T}, T},
        mode::AbstractNonlinearTerminationMode, saved_value_prototype...;
        use_deprecated_retcodes::Val{D} = Val(true),  # Remove in v8, warn in v7
        abstol = nothing, reltol = nothing, kwargs...) where {D, T <: Number}
    abstol = _get_tolerance(abstol, T)
    reltol = _get_tolerance(reltol, T)
    TT = typeof(abstol)
    u_ = mode isa AbstractSafeBestNonlinearTerminationMode ?
         (ArrayInterface.can_setindex(u) ? copy(u) : u) : nothing
    if mode isa AbstractSafeNonlinearTerminationMode
        if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
            initial_objective = __apply_termination_internalnorm(mode.internalnorm, du)
            u0_norm = nothing
        else
            initial_objective = __apply_termination_internalnorm(mode.internalnorm, du) /
                                (__add_and_norm(mode.internalnorm, du, u) + eps(TT))
            u0_norm = mode.max_stalled_steps === nothing ? nothing : norm(u, 2)
        end
        objectives_trace = Vector{TT}(undef, mode.patience_steps)
        step_norm_trace = mode.max_stalled_steps === nothing ? nothing :
                          Vector{TT}(undef, mode.max_stalled_steps)
        best_value = initial_objective
        max_stalled_steps = mode.max_stalled_steps
        if ArrayInterface.can_setindex(u_) && !(u_ isa Number) &&
           step_norm_trace !== nothing
            u_diff_cache = similar(u_)
        else
            u_diff_cache = u_
        end
    else
        initial_objective = nothing
        objectives_trace = nothing
        u0_norm = nothing
        step_norm_trace = nothing
        best_value = __cvt_real(T, Inf)
        max_stalled_steps = nothing
        u_diff_cache = u_
    end

    length(saved_value_prototype) == 0 && (saved_value_prototype = nothing)

    retcode = ifelse(D, NonlinearSafeTerminationReturnCode.Default, ReturnCode.Default)

    return NonlinearTerminationModeCache{D}(u_, retcode, abstol, reltol, best_value, mode,
        initial_objective, objectives_trace, 0, saved_value_prototype, u0_norm,
        step_norm_trace, max_stalled_steps, u_diff_cache)
end

function SciMLBase.reinit!(cache::NonlinearTerminationModeCache{dep_retcode}, du,
        u, saved_value_prototype...; abstol = nothing, reltol = nothing,
        kwargs...) where {dep_retcode}
    T = eltype(cache.abstol)
    length(saved_value_prototype) != 0 && (cache.saved_values = saved_value_prototype)

    u_ = cache.mode isa AbstractSafeBestNonlinearTerminationMode ?
         (ArrayInterface.can_setindex(u) ? copy(u) : u) : nothing
    cache.u = u_
    cache.retcode = ifelse(dep_retcode, NonlinearSafeTerminationReturnCode.Default,
        ReturnCode.Default)

    cache.abstol = _get_tolerance(abstol, T)
    cache.reltol = _get_tolerance(reltol, T)
    cache.nsteps = 0

    mode = get_termination_mode(cache)
    if mode isa AbstractSafeNonlinearTerminationMode
        if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
            initial_objective = __apply_termination_internalnorm(
                cache.mode.internalnorm, du)
        else
            initial_objective = __apply_termination_internalnorm(
                cache.mode.internalnorm, du) /
                                (__add_and_norm(cache.mode.internalnorm, du, u) + eps(TT))
            cache.max_stalled_steps !== nothing && (cache.u0_norm = norm(u_, 2))
        end
        best_value = initial_objective
    else
        initial_objective = nothing
        best_value = __cvt_real(T, Inf)
    end
    cache.best_objective_value = best_value
    cache.initial_objective = initial_objective
    return cache
end

# This dispatch is needed based on how Terminating Callback works!
# This intentially drops the `abstol` and `reltol` arguments
function (cache::NonlinearTerminationModeCache)(integrator::DiffEqBase.AbstractODEIntegrator,
        abstol::Number, reltol::Number, min_t)
    retval = cache(cache.mode, get_du(integrator), integrator.u, integrator.uprev)
    (min_t === nothing || integrator.t ≥ min_t) && return retval
    return false
end
function (cache::NonlinearTerminationModeCache)(du, u, uprev, args...)
    return cache(cache.mode, du, u, uprev, args...)
end

function (cache::NonlinearTerminationModeCache)(mode::AbstractNonlinearTerminationMode, du,
        u, uprev, args...)
    return check_convergence(mode, du, u, uprev, cache.abstol, cache.reltol)
end

function (cache::NonlinearTerminationModeCache{dep_retcode})(
        mode::AbstractSafeNonlinearTerminationMode,
        du, u, uprev, args...) where {dep_retcode}
    if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
        objective = __apply_termination_internalnorm(mode.internalnorm, du)
        criteria = cache.abstol
    else
        objective = __apply_termination_internalnorm(mode.internalnorm, du) /
                    (__add_and_norm(mode.internalnorm, du, u) + eps(cache.abstol))
        criteria = cache.reltol
    end

    # Protective Break
    if isinf(objective) || isnan(objective)
        cache.retcode = ifelse(dep_retcode,
            NonlinearSafeTerminationReturnCode.ProtectiveTermination, ReturnCode.Unstable)
        return true
    end
    ## By default we turn this off since it has the potential for false positives
    if cache.mode.protective_threshold !== nothing &&
       (objective > cache.initial_objective * cache.mode.protective_threshold * length(du))
        cache.retcode = ifelse(dep_retcode,
            NonlinearSafeTerminationReturnCode.ProtectiveTermination, ReturnCode.Unstable)
        return true
    end

    # Check if best solution
    if mode isa AbstractSafeBestNonlinearTerminationMode &&
       objective < cache.best_objective_value
        cache.best_objective_value = objective
        __update_u!!(cache, u)
        if cache.saved_values !== nothing && length(args) ≥ 1
            cache.saved_values = args
        end
    end

    # Main Termination Condition
    if objective ≤ criteria
        cache.retcode = ifelse(dep_retcode,
            NonlinearSafeTerminationReturnCode.Success, ReturnCode.Success)
        return true
    end

    # Terminate if there has been no improvement for the last `patience_steps`
    cache.nsteps += 1
    cache.nsteps == 1 && (cache.initial_objective = objective)
    cache.objectives_trace[mod1(cache.nsteps, length(cache.objectives_trace))] = objective

    if objective ≤ cache.mode.patience_objective_multiplier * criteria
        if cache.nsteps ≥ cache.mode.patience_steps
            if cache.nsteps < length(cache.objectives_trace)
                min_obj, max_obj = extrema(@view(cache.objectives_trace[1:(cache.nsteps)]))
            else
                min_obj, max_obj = extrema(cache.objectives_trace)
            end
            if min_obj < cache.mode.min_max_factor * max_obj
                cache.retcode = ifelse(dep_retcode,
                    NonlinearSafeTerminationReturnCode.PatienceTermination,
                    ReturnCode.Stalled)
                return true
            end
        end
    end

    # Test for stalling if that is not disabled
    if cache.step_norm_trace !== nothing
        if ArrayInterface.can_setindex(cache.u_diff_cache) && !(u isa Number)
            @. cache.u_diff_cache = u - uprev
        else
            cache.u_diff_cache = u .- uprev
        end
        du_norm = norm(cache.u_diff_cache, 2)
        cache.step_norm_trace[mod1(cache.nsteps, length(cache.step_norm_trace))] = du_norm
        if cache.nsteps ≥ cache.mode.max_stalled_steps
            max_step_norm = maximum(cache.step_norm_trace)
            if cache.mode isa AbsSafeTerminationMode ||
               cache.mode isa AbsSafeBestTerminationMode
                stalled_step = max_step_norm ≤ cache.abstol
            else
                stalled_step = max_step_norm ≤
                               cache.reltol * (max_step_norm + cache.u0_norm)
            end
            if stalled_step
                cache.retcode = ifelse(dep_retcode,
                    NonlinearSafeTerminationReturnCode.PatienceTermination,
                    ReturnCode.Stalled)
                return true
            end
        end
    end

    cache.retcode = ifelse(dep_retcode,
        NonlinearSafeTerminationReturnCode.Failure, ReturnCode.Failure)
    return false
end

# Check Convergence
function check_convergence(::SteadyStateDiffEqTerminationMode, duₙ, uₙ, uₙ₋₁, abstol,
        reltol)
    if __fast_scalar_indexing(duₙ, uₙ)
        return all(@closure(xy->begin
                x, y = xy
                return (abs(x) ≤ abstol) | (abs(x) ≤ reltol * abs(y))
            end),
            zip(duₙ, uₙ))
    else
        return all(@. (abs(duₙ) ≤ abstol) | (abs(duₙ) ≤ reltol * abs(uₙ)))
    end
end

function check_convergence(
        ::SimpleNonlinearSolveTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    if __fast_scalar_indexing(duₙ, uₙ)
        return all(@closure(xy->begin
                x, y = xy
                return (abs(x) ≤ abstol) | (abs(x) ≤ reltol * abs(y))
            end),
            zip(duₙ, uₙ)) ||
               __nonlinearsolve_is_approx(uₙ, uₙ₋₁; atol = abstol, rtol = reltol)
    else
        return all(@. (abs(duₙ) ≤ abstol) | (abs(duₙ) ≤ reltol * abs(uₙ))) ||
               __nonlinearsolve_is_approx(uₙ, uₙ₋₁; atol = abstol, rtol = reltol)
    end
end
function check_convergence(::RelTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    if __fast_scalar_indexing(duₙ, uₙ)
        return all(@closure(xy->begin
                x, y = xy
                return abs(x) ≤ reltol * abs(y)
            end), zip(duₙ, uₙ))
    else
        return all(@. abs(duₙ) ≤ reltol * abs(uₙ + duₙ))
    end
end
function check_convergence(::AbsTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    return all(@closure(x->abs(x) ≤ abstol), duₙ)
end

function check_convergence(mode::NormTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    du_norm = __apply_termination_internalnorm(mode.internalnorm, duₙ)
    return (du_norm ≤ abstol) ||
           (du_norm ≤ reltol * __add_and_norm(mode.internalnorm, duₙ, uₙ))
end
function check_convergence(
        mode::Union{
            RelNormTerminationMode, RelSafeTerminationMode, RelSafeBestTerminationMode},
        duₙ, uₙ, uₙ₋₁, abstol, reltol)
    return __apply_termination_internalnorm(mode.internalnorm, duₙ) ≤
           reltol * __add_and_norm(mode.internalnorm, duₙ, uₙ)
end
function check_convergence(
        mode::Union{AbsNormTerminationMode, AbsSafeTerminationMode,
            AbsSafeBestTerminationMode},
        duₙ, uₙ, uₙ₋₁, abstol, reltol)
    return __apply_termination_internalnorm(mode.internalnorm, duₙ) ≤ abstol
end

@inline function __apply_termination_internalnorm(::Nothing, u)
    return __apply_termination_internalnorm(Base.Fix1(maximum, abs), u)
end
@inline __apply_termination_internalnorm(f::F, u) where {F} = f(u)

# Nonlinear Solve Norm (norm(_, 2))
NONLINEARSOLVE_DEFAULT_NORM(u::Union{AbstractFloat, Complex}) = @fastmath abs(u)
function NONLINEARSOLVE_DEFAULT_NORM(f::F,
        u::Union{AbstractFloat, Complex}) where {F}
    return @fastmath abs(f(u))
end

function NONLINEARSOLVE_DEFAULT_NORM(u::Array{
        T}) where {T <: Union{AbstractFloat, Complex}}
    x = zero(T)
    @inbounds @fastmath for ui in u
        x += abs2(ui)
    end
    return Base.FastMath.sqrt_fast(real(x))
end

function NONLINEARSOLVE_DEFAULT_NORM(f::F,
        u::Union{Array{T}, Iterators.Zip{<:Tuple{Vararg{Array{T}}}}}) where {
        F, T <: Union{AbstractFloat, Complex}}
    x = zero(T)
    @inbounds @fastmath for ui in u
        x += abs2(f(ui))
    end
    return Base.FastMath.sqrt_fast(real(x))
end

function NONLINEARSOLVE_DEFAULT_NORM(u::StaticArray{
        <:Tuple, T}) where {T <: Union{AbstractFloat, Complex}}
    return Base.FastMath.sqrt_fast(real(sum(abs2, u)))
end

function NONLINEARSOLVE_DEFAULT_NORM(f::F,
        u::StaticArray{<:Tuple, T}) where {
        F, T <: Union{AbstractFloat, Complex}}
    return Base.FastMath.sqrt_fast(real(sum(abs2 ∘ f, u)))
end

function NONLINEARSOLVE_DEFAULT_NORM(u::AbstractArray)
    return Base.FastMath.sqrt_fast(UNITLESS_ABS2(u))
end

function NONLINEARSOLVE_DEFAULT_NORM(f::F, u::AbstractArray) where {F}
    return Base.FastMath.sqrt_fast(UNITLESS_ABS2(f, u))
end

NONLINEARSOLVE_DEFAULT_NORM(u) = norm(u)
NONLINEARSOLVE_DEFAULT_NORM(f::F, u) where {F} = norm(f.(u))
