function solve(
        prob::NonlinearProblem{<:Union{Number, <:AbstractArray},
            iip, <:Union{
                <:ForwardDiff.Dual{T, V, P}, <:AbstractArray{<:ForwardDiff.Dual{T, V, P}}}},
        alg::AbstractSimpleNonlinearSolveAlgorithm, args...; kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

function solve(
        prob::NonlinearLeastSquaresProblem{<:AbstractArray,
            iip, <:Union{<:AbstractArray{<:ForwardDiff.Dual{T, V, P}}}},
        alg::AbstractSimpleNonlinearSolveAlgorithm, args...; kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

for algType in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder)
    @eval begin
        function SciMLBase.solve(
                prob::IntervalNonlinearProblem{uType, iip,
                    <:Union{<:ForwardDiff.Dual{T, V, P},
                        <:AbstractArray{<:ForwardDiff.Dual{T, V, P}}}},
                alg::$(algType), args...; kwargs...) where {uType, T, V, P, iip}
            sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
            dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
            return SciMLBase.build_solution(prob, alg, dual_soln, sol.resid; sol.retcode,
                sol.stats, sol.original, left = ForwardDiff.Dual{T, V, P}(
                    sol.left, partials),
                right = ForwardDiff.Dual{T, V, P}(sol.right, partials))
        end
    end
end

function __nlsolve_ad(
        prob::Union{IntervalNonlinearProblem, NonlinearProblem}, alg, args...; kwargs...)
    p = SimpleNonlinearSolve.value(prob.p)
    if prob isa IntervalNonlinearProblem
        tspan = SimpleNonlinearSolve.value.(prob.tspan)
        newprob = IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        u0 = SimpleNonlinearSolve.value(prob.u0)
        newprob = NonlinearProblem(prob.f, u0, p; prob.kwargs...)
    end

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    f_p = __nlsolve_∂f_∂p(prob, prob.f, uu, p)
    f_x = __nlsolve_∂f_∂u(prob, prob.f, uu, p)

    z_arr = -f_x \ f_p

    pp = prob.p
    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if uu isa Number
        partials = sum(sumfun, zip(z_arr, pp))
    elseif p isa Number
        partials = sumfun((z_arr, pp))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), pp))
    end

    return sol, partials
end

function __nlsolve_ad(prob::NonlinearLeastSquaresProblem, alg, args...; kwargs...)
    p = SimpleNonlinearSolve.value(prob.p)
    u0 = SimpleNonlinearSolve.value(prob.u0)
    newprob = NonlinearLeastSquaresProblem(prob.f, u0, p; prob.kwargs...)

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u

    # First check for custom `vjp` then custom `Jacobian` and if nothing is provided use
    # nested autodiff as the last resort
    if SciMLBase.has_vjp(prob.f)
        if SciMLBase.isinplace(prob)
            _F = @closure (du, u, p) -> begin
                resid = similar(du, length(sol.resid))
                prob.f(resid, u, p)
                prob.f.vjp(du, resid, u, p)
                du .*= 2
                return nothing
            end
        else
            _F = @closure (u, p) -> begin
                resid = prob.f(u, p)
                return reshape(2 .* prob.f.vjp(resid, u, p), size(u))
            end
        end
    elseif SciMLBase.has_jac(prob.f)
        if SciMLBase.isinplace(prob)
            _F = @closure (du, u, p) -> begin
                J = similar(du, length(sol.resid), length(u))
                prob.f.jac(J, u, p)
                resid = similar(du, length(sol.resid))
                prob.f(resid, u, p)
                mul!(reshape(du, 1, :), vec(resid)', J, 2, false)
                return nothing
            end
        else
            _F = @closure (u, p) -> begin
                return reshape(2 .* vec(prob.f(u, p))' * prob.f.jac(u, p), size(u))
            end
        end
    else
        if SciMLBase.isinplace(prob)
            _F = @closure (du, u, p) -> begin
                resid = similar(du, length(sol.resid))
                res = DiffResults.DiffResult(
                    resid, similar(du, length(sol.resid), length(u)))
                _f = @closure (du, u) -> prob.f(du, u, p)
                ForwardDiff.jacobian!(res, _f, resid, u)
                mul!(reshape(du, 1, :), vec(DiffResults.value(res))',
                    DiffResults.jacobian(res), 2, false)
                return nothing
            end
        else
            # For small problems, nesting ForwardDiff is actually quite fast
            if SimpleNonlinearSolve.__is_extension_loaded(Val(:Zygote)) &&
               (length(uu) + length(sol.resid) ≥ 50)
                _F = @closure (u, p) -> SimpleNonlinearSolve.__zygote_compute_nlls_vjp(
                    prob.f, u, p)
            else
                _F = @closure (u, p) -> begin
                    T = promote_type(eltype(u), eltype(p))
                    res = DiffResults.DiffResult(
                        similar(u, T, size(sol.resid)), similar(
                            u, T, length(sol.resid), length(u)))
                    ForwardDiff.jacobian!(res, Base.Fix2(prob.f, p), u)
                    return reshape(
                        2 .* vec(DiffResults.value(res))' * DiffResults.jacobian(res),
                        size(u))
                end
            end
        end
    end

    f_p = __nlsolve_∂f_∂p(prob, _F, uu, p)
    f_x = __nlsolve_∂f_∂u(prob, _F, uu, p)

    z_arr = -f_x \ f_p

    pp = prob.p
    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if uu isa Number
        partials = sum(sumfun, zip(z_arr, pp))
    elseif p isa Number
        partials = sumfun((z_arr, pp))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), pp))
    end

    return sol, partials
end

@inline function __nlsolve_∂f_∂p(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        __f = p -> begin
            du = similar(u, promote_type(eltype(u), eltype(p)))
            f(du, u, p)
            return du
        end
    else
        __f = Base.Fix1(f, u)
    end
    if p isa Number
        return SimpleNonlinearSolve.__reshape(ForwardDiff.derivative(__f, p), :, 1)
    elseif u isa Number
        return SimpleNonlinearSolve.__reshape(ForwardDiff.gradient(__f, p), 1, :)
    else
        return ForwardDiff.jacobian(__f, p)
    end
end

@inline function __nlsolve_∂f_∂u(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        du = similar(u)
        __f = (du, u) -> f(du, u, p)
        ForwardDiff.jacobian(__f, du, u)
    else
        __f = Base.Fix2(f, p)
        if u isa Number
            return ForwardDiff.derivative(__f, u)
        else
            return ForwardDiff.jacobian(__f, u)
        end
    end
end

@inline function __nlsolve_dual_soln(u::Number, partials,
        ::Union{<:AbstractArray{<:ForwardDiff.Dual{T, V, P}}, ForwardDiff.Dual{T, V, P}}) where {
        T, V, P}
    return ForwardDiff.Dual{T, V, P}(u, partials)
end

@inline function __nlsolve_dual_soln(u::AbstractArray,
        partials,
        ::Union{<:AbstractArray{<:ForwardDiff.Dual{T, V, P}}, ForwardDiff.Dual{T, V, P}}) where {
        T, V, P}
    _partials = SimpleNonlinearSolve._restructure(u, partials)
    return map(((uᵢ, pᵢ),) -> ForwardDiff.Dual{T, V, P}(uᵢ, pᵢ), zip(u, _partials))
end
