module SimpleNonlinearSolvePolyesterForwardDiffExt

import ADTypes: AutoPolyesterForwardDiff
import SimpleNonlinearSolve, PolyesterForwardDiff

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:PolyesterForwardDiff}) = true

function SimpleNonlinearSolve.__value_and_jacobian!(
        ::Val{iip}, ad::AutoPolyesterForwardDiff, J, f::F, y,
        x::AbstractArray, cache) where {iip, F}
    if iip
        PolyesterForwardDiff.threaded_jacobian!(f, y, J, x, cache)
        f(y, x)
        return y, J
    end
    return f(x), PolyesterForwardDiff.threaded_jacobian!(f, J, x, cache)
end

end
