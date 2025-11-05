#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    KirlikSayin()

`KirlikSayin` implements the algorithm of:

Kirlik, G., & Sayın, S. (2014). A new algorithm for generating all nondominated
solutions of multiobjective discrete optimization problems. European Journal of
Operational Research, 232(3), 479-488.

This is an algorithm to generate all nondominated solutions for multi-objective
discrete optimization problems. The algorithm maintains `(p-1)`-dimensional
rectangle regions in the solution space, and a two-stage optimization problem
is solved for each rectangle.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
    list of current solutions.
"""
struct KirlikSayinParallel <: AbstractAlgorithm end

struct _RectangleParallel
    l::Vector{Float64}
    u::Vector{Float64}

    function _RectangleParallel(l::Vector{Float64}, u::Vector{Float64})
        @assert length(l) == length(u) "Dimension mismatch between l and u"
        return new(l, u)
    end
end

_volume(r::_RectangleParallel, l::Vector{Float64}) = prod(r.u - l)

function Base.issubset(x::_RectangleParallel, y::_RectangleParallel)
    @assert length(x.l) == length(y.l) "Dimension mismatch"
    return all(x.l .>= y.l) && all(x.u .<= y.u)
end

function _remove_RectangleParallel(L::Vector{_RectangleParallel}, R::_RectangleParallel)
    index_to_remove = Int[t for (t, x) in enumerate(L) if issubset(x, R)]
    deleteat!(L, index_to_remove)
    return
end

function _split_RectangleParallel(r::_RectangleParallel, axis::Int, f::Float64)
    l = [i != axis ? r.l[i] : f for i in 1:length(r.l)]
    u = [i != axis ? r.u[i] : f for i in 1:length(r.l)]
    return _RectangleParallel(r.l, u), _RectangleParallel(l, r.u)
end

function _update_list(L::Vector{_RectangleParallel}, f::Vector{Float64})
    L_new = _RectangleParallel[]
    for Rᵢ in L
        lᵢ, uᵢ = Rᵢ.l, Rᵢ.u
        T = [Rᵢ]
        for j in 1:length(f)
            if lᵢ[j] < f[j] < uᵢ[j]
                T̄ = _RectangleParallel[]
                for Rₜ in T
                    a, b = _split_RectangleParallel(Rₜ, j, f[j])
                    push!(T̄, a)
                    push!(T̄, b)
                end
                T = T̄
            end
        end
        append!(L_new, T)
    end
    return L_new
end

function rebuild_model(model::Optimizer)::Optimizer
    new_model = Optimizer(model.optimizer_factory)
    MOI.copy_to(new_model, model.inner)
    
    # silent = MOI.get(model.inner, MOI.Silent())
   MOI.set(new_model, MOI.Silent(), true)

    return new_model
end

function minimize_multiobjective!(algorithm::KirlikSayinParallel, model::Optimizer)
    @assert MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    start_time = time()    

    solutions = SolutionPoint[]
    # Problem with p objectives.
    # Set k = 1, meaning the nondominated points will get projected
    # down to the objective {2, 3, ..., p}
    k = 1
    YN = Vector{Float64}[]
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(model.f)
    #yI, yN = zeros(n), zeros(n)
    sharedIdealPoint = SharedArrays.SharedVector{Float64}(n)
    yI = SharedArrays.SharedVector{Float64}(n)
    yN = SharedArrays.SharedVector{Float64}(n)
    # This tolerance is really important!
    δ = 1.0
    scalars = MOI.Utilities.scalarize(model.f)

    # Ideal and Nadir point estimation
    results = Distributed.pmap(i -> begin
        f_i = scalars[i]
        local_model = rebuild_model(model)

        # Ideal point
        MOI.set(local_model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        optimize_inner!(local_model)
        status = MOI.get(local_model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        _, Y = _compute_point(local_model, variables, f_i)
        sharedIdealPoint[i] = yI[i] = Y
        
        # Nadir point
        MOI.set(local_model.inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        optimize_inner!(local_model)
        status = MOI.get(local_model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            # Repair ObjectiveSense before exiting
            MOI.set(local_model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            _warn_on_nonfinite_anti_ideal(algorithm, MOI.MIN_SENSE, i)
            return status, nothing
        end
        _, Y = _compute_point(local_model, variables, f_i)
        yN[i] = Y + δ
        MOI.set(local_model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end, 1:length(scalars))
    
    model.ideal_point = collect(sharedIdealPoint)
    LShared = SharedArrays.SharedVector{_RectangleParallel}(1)
    LShared[1] = _RectangleParallel(_project(yI, k), _project(yN, k))
    L = [_RectangleParallel(_project(yI, k), _project(yN, k))]
    status = MOI.OPTIMAL
    while !isempty(L)
        if (ret = _check_premature_termination(model, start_time)) !== nothing
            status = ret
            break
        end

        volumns_to_grap = min(LShared.length, Distributed.nworkers())
        max_volume_indexs = sort(collect(1:length(L)), by = i -> _volume(L[i], _project(yI, k)), rev = true)[1:volumns_to_grap]

        result = Distributed.pmap(max_volume_index -> begin 
            uᵢ = LShared[max_volume_index].u
            local_model = rebuild_model(model)

            # Solving the first stage model: P_k(ε)
            #   minimize: f_1(x)
            #       s.t.: f_i(x) <= u_i - δ
            @assert k == 1
            MOI.set(
                local_model.inner,
                MOI.ObjectiveFunction{typeof(scalars[k])}(),
                scalars[k],
            )
            ε_constraints = Any[]
            for (i, f_i) in enumerate(scalars)
                if i == k
                    continue
                end
                ci = MOI.Utilities.normalize_and_add_constraint(
                    local_model.inner,
                    f_i,
                    MOI.LessThan{Float64}(uᵢ[i-1] - δ),
                )
                push!(ε_constraints, ci)
            end
            optimize_inner!(local_model)
            if !_is_scalar_status_optimal(local_model)
                # If this fails, it likely means that the solver experienced a
                # numerical error with this box. Just skip it.
                _remove_RectangleParallel(L, _RectangleParallel(_project(yI, k), uᵢ))
                MOI.delete.(local_model, ε_constraints)
                return nothing
            end
            zₖ = MOI.get(local_model.inner, MOI.ObjectiveValue())
            # Solving the second stage local_model: Q_k(ε, zₖ)
            # Set objective sum(local_model.f)
            sum_f = MOI.Utilities.operate(+, Float64, scalars...)
            MOI.set(local_model.inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
            # Constraint to eliminate weak dominance
            zₖ_constraint = MOI.Utilities.normalize_and_add_constraint(
                local_model.inner,
                scalars[k],
                MOI.EqualTo(zₖ),
            )
            optimize_inner!(local_model)
            if !_is_scalar_status_optimal(local_model)
                # If this fails, it likely means that the solver experienced a
                # numerical error with this box. Just skip it.
                MOI.delete.(local_model, ε_constraints)
                MOI.delete(local_model, zₖ_constraint)
                _remove_RectangleParallel(L, _RectangleParallel(_project(yI, k), uᵢ))
                return nothing
            end
            X, Y = _compute_point(local_model, variables, local_model.f)
            Y_proj = _project(Y, k)
            if !(Y in YN)
                push!(solutions, SolutionPoint(X, Y))
                push!(YN, Y)
                L = _update_list(L, Y_proj)
            end
            MOI.delete(local_model, zₖ_constraint)
            MOI.delete.(local_model, ε_constraints)
            _remove_RectangleParallel(L, _RectangleParallel(Y_proj, uᵢ))
    
    end, max_volume_indexs)

    end
    return status, filter_nondominated(MOI.MIN_SENSE, solutions)
end
