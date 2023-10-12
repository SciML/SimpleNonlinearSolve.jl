struct SimpleJNFKJacVecTag end

function jvp_forwarddiff(f, x::AbstractArray{T}, v) where {T}
    v_ = reshape(v, axes(x))
    y = (Dual{Tag{SimpleJNFKJacVecTag, T}, T, 1}).(x, Partials.(tuple.(v_)))
    return vec(ForwardDiff.partials.(vec(f(y)), 1))
end

struct JacVecOperator{F, X}
    f::F
    x::X
end

(jvp::JacVecOperator)(v, _, _) = jvp_forwarddiff(jvp.f, jvp.x, v)

"""
    SimpleJNFK()

"""
struct SimpleJFNK end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleJFNK, args...;
    abstol = nothing, reltol= nothing, maxiters = 1000, linsolve_kwargs = (;), kwargs...)
    iip = SciMLBase.isinplace(prob)
    @assert !iip "SimpleJFNK does not support inplace problems"

    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fx = f(x)
    T = typeof(x)

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    op = FunctionOperator(JacVecOperator(f, x), x)
    linprob = LinearProblem(op, -fx)
    lincache = init(linprob, SimpleGMRES(); abstol, reltol, maxiters, linsolve_kwargs...)

    for i in 1:maxiters
        iszero(fx) &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

        linsol = solve!(lincache)
        x .-= linsol.u
        lincache = linsol.cache

        # FIXME: not nothing
        if isapprox(x, nothing; atol, rtol)
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        end
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
