struct SimpleJNFKJacVecTag end

function jvp_forwarddiff(f, x::AbstractArray{T}, v) where {T}
    v_ = reshape(v, axes(x))
    y = (Dual{Tag{SimpleJNFKJacVecTag, T}, T, 1}).(x, Partials.(tuple.(v_)))
    return vec(ForwardDiff.partials.(vec(f(y)), 1))
end
jvp_forwarddiff!(r, f, x, v) = copyto!(r, jvp_forwarddiff(f, x, v))

struct JacVecOperator{F, X}
    f::F
    x::X
end

(jvp::JacVecOperator)(v, _, _) = jvp_forwarddiff(jvp.f, jvp.x, v)
(jvp::JacVecOperator)(r, v, _, _) = jvp_forwarddiff!(r, jvp.f, jvp.x, v)

"""
    SimpleJNFK(; batched::Bool = false)

A low overhead Jacobian-free Newton-Krylov method. This method internally uses `GMRES` to
avoid computing the Jacobian Matrix.

!!! warning

    JNFK doesn't work well without preconditioning, which is currently not supported. We
    recommend using `NewtonRaphson(linsolve = KrylovJL_GMRES())` for preconditioning
    support.
"""
struct SimpleJFNK end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleJFNK, args...;
    abstol = nothing, reltol = nothing, maxiters = 1000, linsolve_kwargs = (;), kwargs...)
    iip = SciMLBase.isinplace(prob)
    @assert !iip "SimpleJFNK does not support inplace problems"

    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fx = f(x)
    T = typeof(x)

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    op = FunctionOperator(JacVecOperator(f, x), x)
    linprob = LinearProblem(op, vec(fx))
    lincache = init(linprob, KrylovJL_GMRES(); abstol = atol, reltol = rtol, maxiters,
        linsolve_kwargs...)

    for i in 1:maxiters
        linsol = solve!(lincache)
        axpy!(-1, linsol.u, x)
        lincache = linsol.cache

        fx = f(x)

        norm(fx, Inf) ≤ atol &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

        lincache.b = vec(fx)
        lincache.A = FunctionOperator(JacVecOperator(f, x), x)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
