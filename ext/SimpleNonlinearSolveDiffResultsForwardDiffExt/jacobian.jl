
struct SimpleNonlinearSolveTag end

function ForwardDiff.checktag(::Type{<:ForwardDiff.Tag{<:SimpleNonlinearSolveTag, <:T}},
        f::F, x::AbstractArray{T}) where {T, F}
    return true
end

@inline __standard_tag(::Nothing, x) = ForwardDiff.Tag(SimpleNonlinearSolveTag(), eltype(x))
@inline __standard_tag(tag::ForwardDiff.Tag, _) = tag
@inline __standard_tag(tag, x) = ForwardDiff.Tag(tag, eltype(x))

function __pick_forwarddiff_chunk(
        ad::Union{AutoForwardDiff{CS}, AutoPolyesterForwardDiff{CS}}, x) where {CS}
    (CS === nothing || CS ≤ 0) && return __pick_forwarddiff_chunk(x)
    return ForwardDiff.Chunk{CS}()
end
__pick_forwarddiff_chunk(x) = ForwardDiff.Chunk(length(x))
function __pick_forwarddiff_chunk(x::StaticArray)
    L = prod(Size(x))
    if L ≤ ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        return ForwardDiff.Chunk{L}()
    else
        return ForwardDiff.Chunk{ForwardDiff.DEFAULT_CHUNK_THRESHOLD}()
    end
end

# Jacobian
function __forwarddiff_jacobian_config(f::F, x, ck::ForwardDiff.Chunk, tag) where {F}
    return ForwardDiff.JacobianConfig(f, x, ck, tag)
end
function __forwarddiff_jacobian_config(
        f::F, x::SArray, ck::ForwardDiff.Chunk{N}, tag) where {F, N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, eltype(x)})
    duals = ForwardDiff.Dual{typeof(tag), eltype(x), N}.(x)
    return ForwardDiff.JacobianConfig{typeof(tag), eltype(x), N, typeof(duals)}(seeds,
        duals)
end

function __get_jacobian_config(ad::AutoForwardDiff{CS}, f::F, x) where {F, CS}
    return __forwarddiff_jacobian_config(
        f, x, __pick_forwarddiff_chunk(ad, x), __standard_tag(ad.tag, x))
end

function __get_jacobian_config(ad::AutoForwardDiff{CS}, f!::F, y, x) where {F, CS}
    return ForwardDiff.JacobianConfig(
        f!, y, x, __pick_forwarddiff_chunk(ad, x), __standard_tag(ad.tag, x))
end

function __get_jacobian_config(ad::AutoPolyesterForwardDiff{CS}, args...) where {CS}
    return __pick_forwarddiff_chunk(ad, last(args))
end

function SimpleNonlinearSolve.__jacobian_cache(
        ::Val{iip}, ad::Union{AutoForwardDiff, AutoPolyesterForwardDiff}, f::F, y,
        x) where {iip, F}
    if iip
        J = similar(y, promote_type(eltype(x), eltype(y)), length(y), length(x))
        return J, __get_jacobian_config(ad, f, y, x)
    end
    if ad isa AutoPolyesterForwardDiff
        @assert ArrayInterface.can_setindex(x) "PolyesterForwardDiff requires mutable \
                                                inputs. Use AutoForwardDiff instead."
    end
    J = ArrayInterface.can_setindex(x) ?
        similar(y, promote_type(eltype(x), eltype(y)), length(y), length(x)) : nothing
    return J, __get_jacobian_config(ad, f, x)
end

function SimpleNonlinearSolve.__value_and_jacobian!(
        ::Val{iip}, ad::AutoForwardDiff, J, f::F, y, x::AbstractArray, cache) where {iip, F}
    if iip
        res = DiffResults.DiffResult(y, J)
        ForwardDiff.jacobian!(res, f, y, x, cache)
        return DiffResults.value(res), DiffResults.jacobian(res)
    end
    if ArrayInterface.can_setindex(x)
        res = DiffResults.DiffResult(y, J)
        ForwardDiff.jacobian!(res, f, x, cache)
        return DiffResults.value(res), DiffResults.jacobian(res)
    end
    return f(x), ForwardDiff.jacobian(f, x, cache)
end

function SimpleNonlinearSolve.__value_and_jacobian!(
        ::Val, ad::Union{AutoForwardDiff, AutoPolyesterForwardDiff},
        J, f::F, y, x::Number, cache) where {F}
    if hasfield(typeof(ad), :tag)
        T = typeof(__standard_tag(ad.tag, x))
    else
        T = typeof(__standard_tag(nothing, x))
    end
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    return ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end
