module SimpleNonlinearSolveChainRulesCoreDiffEqBaseExt

using SciMLBase
import DiffEqBase, SimpleNonlinearSolve
import ChainRulesCore as CRC

# The expectation here is that no-one is using this directly inside a GPU kernel. We can
# eventually lift this requirement using a custom adjoint
function CRC.rrule(
        ::typeof(SimpleNonlinearSolve.__internal_solve_up), prob::NonlinearProblem,
        sensealg::Union{Nothing, DiffEqBase.AbstractSensitivityAlgorithm}, u0, u0_changed,
        p, p_changed, alg, args...; kwargs...)
    out, ∇internal = DiffEqBase._solve_adjoint(prob, sensealg, u0, p,
        SciMLBase.ChainRulesOriginator(), alg, args...; kwargs...)
    function ∇__internal_solve_up(Δ)
        ∂f, ∂prob, ∂sensealg, ∂u0, ∂p, ∂originator, ∂args... = ∇internal(Δ)
        return (
            ∂f, ∂prob, ∂sensealg, ∂u0, CRC.NoTangent(), ∂p, CRC.NoTangent(), ∂originator,
            ∂args...)
    end
    return out, ∇__internal_solve_up
end

end
