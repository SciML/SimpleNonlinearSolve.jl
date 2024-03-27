module SimpleNonlinearSolveLinearSolveExt

using SimpleNonlinearSolve, LinearSolve

@inline function SimpleNonlinearSolve.__call_KrylovJL_GMRES()
    LinearSolve.KrylovJL_GMRES()
end

end
