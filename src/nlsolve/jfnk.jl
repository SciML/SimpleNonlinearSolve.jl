"""
    SimpleJFNK()
A low overhead Jacobian-free Newton-Krylov method. This method internally uses `GMRES` to
avoid computing the Jacobian Matrix.
!!! warning
    JFNK doesn't work well without preconditioning, which is currently not supported. We
    recommend using `NewtonRaphson(linsolve = KrylovJL_GMRES())` for preconditioning
    support.
"""
@concrete struct SimpleJFNK <: AbstractNewtonAlgorithm
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleJFNK,
        args...; abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, alias_u0 = false, linsolve_kwargs = (;), kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    fx = _get_fx(prob, x)
    @bb xo = copy(x)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    n = length(x)
    ε(u) = 1.0e-6 * (norm(x, 1) + 1) / (n*norm(u))
    jv(u, p, t) = (_get_fx(prob.f, x + ε(u)*u, p) - _get_fx(prob.f, x, p)) / ε(u)
    function jv!(v, u, p, t)
        v .= jv(u, p, t)
        return v
    end

    for i in 1:maxiters

        if i == 1
            iszero(fx) && build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        else
            # Termination Checks
            tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
            tc_sol !== nothing && return tc_sol
        end

        op = FunctionOperator(jv, x)
        linprob = LinearProblem(op, vec(fx))
        lincache = init(linprob, __call_KrylovJL_GMRES(); abstol = abstol, reltol = reltol,
            maxiters = maxiters, linsolve_kwargs...)

        linsol = solve!(lincache)
        lincache = linsol.cache

        @bb copyto!(xo, x)
        δx = linsol.u
        @bb x .-= δx
        fx = _get_fx(prob, x)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
