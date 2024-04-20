"""
    SimpleMuller()

Muller's method.
"""
struct SimpleMuller <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.solve(prob::NonlinearProblem, alg::SimpleMuller, args...;
    abstol = nothing, maxiters = 1000, kwargs...)
    @assert !isinplace(prob) "`SimpleMuller` only supports OOP problems."
    xᵢ₋₂, xᵢ₋₁, xᵢ = prob.u0
    @assert xᵢ₋₂ ≠ xᵢ₋₁ ≠ xᵢ ≠ xᵢ₋₂
    f = Base.Fix2(prob.f, prob.p)
    fxᵢ₋₂, fxᵢ₋₁, fxᵢ = f(xᵢ₋₂), f(xᵢ₋₁), f(xᵢ)

    abstol = __get_tolerance(nothing, abstol,
            promote_type(eltype(fxᵢ₋₂), eltype(xᵢ₋₂)))

    for _ ∈ 1:maxiters
        q = (xᵢ - xᵢ₋₁)/(xᵢ₋₁ - xᵢ₋₂)
        A = q*fxᵢ - q*(1 + q)*fxᵢ₋₁ + q^2*fxᵢ₋₂
        B = (2*q + 1)*fxᵢ - (1 + q)^2*fxᵢ₋₁ + q^2*fxᵢ₋₂
        C = (1 + q)*fxᵢ

        denom₊ = B + √(B^2 - 4*A*C)
        denom₋ = B - √(B^2 - 4*A*C)

        if abs(denom₊) ≥ abs(denom₋)
            xᵢ₊₁ = xᵢ - (xᵢ - xᵢ₋₁)*2*C/denom₊
        else
            xᵢ₊₁ = xᵢ - (xᵢ - xᵢ₋₁)*2*C/denom₋
        end

        fxᵢ₊₁ = f(xᵢ₊₁)

        # Termination Check
        if abstol ≥ abs(fxᵢ₊₁)
            return build_solution(prob, alg, xᵢ₊₁, fxᵢ₊₁;
                                  retcode = ReturnCode.Success)
        end

        xᵢ₋₂, xᵢ₋₁, xᵢ = xᵢ₋₁, xᵢ, xᵢ₊₁
        fxᵢ₋₂, fxᵢ₋₁, fxᵢ = fxᵢ₋₁, fxᵢ, fxᵢ₊₁
    end

    return build_solution(prob, alg, xᵢ₊₁, fxᵢ₊₁;
                          retcode = ReturnCode.MaxIters)
end
