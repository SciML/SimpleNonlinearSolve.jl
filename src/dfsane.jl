"""
    SimpleDFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0,
        M::Int = 10, γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
        nexp::Int = 2, η_strategy::Function = (f_1, k, x, F) -> f_1 ./ k^2,
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing),
        batched::Bool = false,
        max_inner_iterations::Int = 1000)

A low-overhead implementation of the df-sane method for solving large-scale nonlinear
systems of equations. For in depth information about all the parameters and the algorithm,
see the paper: [W LaCruz, JM Martinez, and M Raydan (2006), Spectral residual mathod without
gradient information for solving large-scale nonlinear systems of equations, Mathematics of
Computation, 75, 1429-1448.](https://www.researchgate.net/publication/220576479_Spectral_Residual_Method_without_Gradient_Information_for_Solving_Large-Scale_Nonlinear_Systems_of_Equations)

### Keyword Arguments

- `σ_min`: the minimum value of the spectral coefficient `σ_k` which is related to the step
  size in the algorithm. Defaults to `1e-10`.
- `σ_max`: the maximum value of the spectral coefficient `σ_k` which is related to the step
  size in the algorithm. Defaults to `1e10`.
- `σ_1`: the initial value of the spectral coefficient `σ_k` which is related to the step
  size in the algorithm.. Defaults to `1.0`.
- `M`: The monotonicity of the algorithm is determined by a this positive integer.
  A value of 1 for `M` would result in strict monotonicity in the decrease of the L2-norm
  of the function `f`. However, higher values allow for more flexibility in this reduction.
  Despite this, the algorithm still ensures global convergence through the use of a
  non-monotone line-search algorithm that adheres to the Grippo-Lampariello-Lucidi
  condition. Values in the range of 5 to 20 are usually sufficient, but some cases may call
  for a higher value of `M`. The default setting is 10.
- `γ`: a parameter that influences if a proposed step will be accepted. Higher value of `γ`
  will make the algorithm more restrictive in accepting steps. Defaults to `1e-4`.
- `τ_min`: if a step is rejected the new step size will get multiplied by factor, and this
  parameter is the minimum value of that factor. Defaults to `0.1`.
- `τ_max`: if a step is rejected the new step size will get multiplied by factor, and this
  parameter is the maximum value of that factor. Defaults to `0.5`.
- `nexp`: the exponent of the loss, i.e. ``f_k=||F(x_k)||^{nexp}``. The paper uses
  `nexp ∈ {1,2}`. Defaults to `2`.
- `η_strategy`:  function to determine the parameter `η_k`, which enables growth
  of ``||F||^2``. Called as ``η_k = η_strategy(f_1, k, x, F)`` with `f_1` initialized as
  ``f_1=||F(x_1)||^{nexp}``, `k` is the iteration number, `x` is the current `x`-value and
  `F` the current residual. Should satisfy ``η_k > 0`` and ``∑ₖ ηₖ < ∞``. Defaults to
  ``||F||^2 / k^2``.
- `termination_condition`: a `NLSolveTerminationCondition` that determines when the solver
  should terminate. Defaults to `NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
  abstol = nothing, reltol = nothing)`.
- `batched`: if `true`, the algorithm will use a batched version of the algorithm that treats each
  column of `x` as a separate problem. This can be useful nonlinear problems involing neural
  networks. Defaults to `false`.
- `max_inner_iterations`: the maximum number of iterations allowed for the inner loop of the
  algorithm. Used exclusively in `batched` mode. Defaults to `1000`.
"""
struct SimpleDFSane{T, TC} <: AbstractSimpleNonlinearSolveAlgorithm
    σ_min::T
    σ_max::T
    σ_1::T
    M::Int
    γ::T
    τ_min::T
    τ_max::T
    nexp::Int
    η_strategy::Function
    termination_condition::TC
end

function SimpleDFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0,
        M::Int = 10, γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
        nexp::Int = 2, η_strategy::Function = (f_1, k, x, F) -> f_1 ./ k^2,
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing),
        batched::Bool = false,
        max_inner_iterations = 1000)
    if batched
        return BatchedSimpleDFSane(; σₘᵢₙ = σ_min,
            σₘₐₓ = σ_max,
            σ₁ = σ_1,
            M,
            γ,
            τₘᵢₙ = τ_min,
            τₘₐₓ = τ_max,
            nₑₓₚ = nexp,
            ηₛ = η_strategy,
            termination_condition,
            max_inner_iterations)
    end
    return SimpleDFSane{typeof(σ_min), typeof(termination_condition)}(σ_min,
        σ_max,
        σ_1,
        M,
        γ,
        τ_min,
        τ_max,
        nexp,
        η_strategy,
        termination_condition)
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleDFSane,
        args...; abstol = nothing, reltol = nothing, maxiters = 1000,
        kwargs...)
    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)

    T = eltype(x)
    σ_min = float(alg.σ_min)
    σ_max = float(alg.σ_max)
    σ_k = float(alg.σ_1)

    M = alg.M
    γ = float(alg.γ)
    τ_min = float(alg.τ_min)
    τ_max = float(alg.τ_max)
    nexp = alg.nexp
    η_strategy = alg.η_strategy

    if SciMLBase.isinplace(prob)
        error("SimpleDFSane currently only supports out-of-place nonlinear problems")
    end

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        error("SimpleDFSane currently doesn't support SAFE_BEST termination modes")
    end

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
              nothing
    termination_condition = tc(storage)

    function ff(x)
        F = f(x)
        f_k = norm(F)^nexp
        return f_k, F
    end

    function generate_history(f_k, M)
        return fill(f_k, M)
    end

    f_k, F_k = ff(x)
    α_1 = convert(T, 1.0)
    f_1 = f_k
    history_f_k = generate_history(f_k, M)

    for k in 1:maxiters
        # Spectral parameter range check
        σ_k = sign(σ_k) * clamp(abs(σ_k), σ_min, σ_max)

        # Line search direction
        d = -σ_k .* F_k

        η = η_strategy(f_1, k, x, F_k)
        f̄ = maximum(history_f_k)
        α_p = α_1
        α_m = α_1
        x_new = @. x + α_p * d

        f_new, F_new = ff(x_new)

        inner_iterations = 0
        while true
            inner_iterations += 1

            criteria = f̄ + η - γ * α_p^2 * f_k
            f_new ≤ criteria && break

            α_tp = @. α_p^2 * f_k / (f_new + (2 * α_p - 1) * f_k)
            x_new = @. x - α_m * d
            f_new, F_new = ff(x_new)

            f_new ≤ criteria && break

            α_tm = @. α_m^2 * f_k / (f_new + (2 * α_m - 1) * f_k)
            α_p = @. clamp(α_tp, τ_min * α_p, τ_max * α_p)
            α_m = @. clamp(α_tm, τ_min * α_m, τ_max * α_m)
            x_new = @. x + α_p * d
            f_new, F_new = ff(x_new)
        end

        if termination_condition(F_new, x_new, x, atol, rtol)
            return SciMLBase.build_solution(prob,
                alg,
                x_new,
                F_new;
                retcode = ReturnCode.Success)
        end

        # Update spectral parameter
        s_k = @. x_new - x
        y_k = @. F_new - F_k

        σ_k = (s_k' * s_k) / (s_k' * y_k)

        # Take step
        x = x_new
        F_k = F_new
        f_k = f_new

        # Store function value
        history_f_k[k % M + 1] = f_new
    end
    return SciMLBase.build_solution(prob, alg, x, F_k; retcode = ReturnCode.MaxIters)
end
