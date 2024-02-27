function SimpleNonlinearSolve.compute_jacobian_and_hessian(
        ad::AutoForwardDiff, prob, _, x::Number)
    fx = prob.f(x, prob.p)
    J_fn = Base.Fix1(ForwardDiff.derivative, Base.Fix2(prob.f, prob.p))
    dfx = J_fn(x)
    d2fx = ForwardDiff.derivative(J_fn, x)
    return fx, dfx, d2fx
end

function SimpleNonlinearSolve.compute_jacobian_and_hessian(
        ad::AutoForwardDiff, prob, fx, x)
    if SciMLBase.isinplace(prob)
        error("Inplace version for Nested ForwardDiff Not Implemented Yet!")
    else
        f = Base.Fix2(prob.f, prob.p)
        fx = f(x)
        J_fn = Base.Fix1(ForwardDiff.jacobian, f)
        dfx = J_fn(x)
        d2fx = ForwardDiff.jacobian(J_fn, x)
        return fx, dfx, d2fx
    end
end
