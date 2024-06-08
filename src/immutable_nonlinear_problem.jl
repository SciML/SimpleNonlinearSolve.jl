struct ImmutableNonlinearProblem{uType, isinplace, P, F, K, PT} <:
       AbstractNonlinearProblem{uType, isinplace}
    f::F
    u0::uType
    p::P
    problem_type::PT
    kwargs::K
    @add_kwonly function ImmutableNonlinearProblem{iip}(f::AbstractNonlinearFunction{iip}, u0,
            p = NullParameters(),
            problem_type = StandardNonlinearProblem();
            kwargs...) where {iip}
        if haskey(kwargs, :p)
            error("`p` specified as a keyword argument `p = $(kwargs[:p])` to `NonlinearProblem`. This is not supported.")
        end
        warn_paramtype(p)
        new{typeof(u0), iip, typeof(p), typeof(f),
            typeof(kwargs), typeof(problem_type)}(f,
            u0,
            p,
            problem_type,
            kwargs)
    end

    """

    Define a steady state problem using the given function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function ImmutableNonlinearProblem{iip}(f, u0, p = NullParameters(); kwargs...) where {iip}
        ImmutableNonlinearProblem{iip}(NonlinearFunction{iip}(f), u0, p; kwargs...)
    end
end

"""

Define a nonlinear problem using an instance of
[`AbstractNonlinearFunction`](@ref AbstractNonlinearFunction).
"""
function ImmutableNonlinearProblem(f::AbstractNonlinearFunction, u0, p = NullParameters(); kwargs...)
    ImmutableNonlinearProblem{isinplace(f)}(f, u0, p; kwargs...)
end

function ImmutableNonlinearProblem(f, u0, p = NullParameters(); kwargs...)
    ImmutableNonlinearProblem(NonlinearFunction(f), u0, p; kwargs...)
end

"""

Define a ImmutableNonlinearProblem problem from SteadyStateProblem
"""
function ImmutableNonlinearProblem(prob::AbstractNonlinearProblem)
    ImmutableNonlinearProblem{isinplace(prob)}(prob.f, prob.u0, prob.p)
end

staticarray_itize(x) = x
staticarray_itize(x::Vector) = SVector{length(x)}(x)
staticarray_itize(x::SizedVector) = SVector{length(x)}(x)
staticarray_itize(x::Matrix) = SMatrix{size(x)...}(x)
staticarray_itize(x::SizedMatrix) = SMatrix{size(x)...}(x)

function Base.convert(::Type{ImmutableNonlinearProblem}, prob::T) where {T <: NonlinearProblem}
    ImmutableNonlinearProblem{isinplace(prob)}(prob.f,
        staticarray_itize(prob.u0),
        staticarray_itize(prob.p),
        prob.problem_type;
        prob.kwargs...)
end
