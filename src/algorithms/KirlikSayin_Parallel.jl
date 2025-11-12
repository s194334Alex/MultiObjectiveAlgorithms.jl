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

struct KirlikSayinParallelInfo
    scales::Vector{MOI.ScalarAffineFunction{Float64}}
    model::Optimizer
    start_time::Float64
    variables::Vector{MOI.VariableIndex}
    algorithm::KirlikSayinParallel
end

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
    new_model.f = model.f
    

    MOI.set(new_model, MOI.Silent(), true)

    return new_model
end

function _distribute_info_to_workers!(info::KirlikSayinParallelInfo)
    @assert Distributed.nprocs() > 1 "At least 2 processes are required for parallel execution"
    for p in Distributed.workers()
        Distributed.remotecall_fetch(p) do
            global SCALARS = info.scales
            global LOCAL_MODEL = copy(info.model)
            global START_TIME = info.start_time
            global K = 1
            global δ = 1.0
            global VARIABLES = info.variables
            global ALGO = info.algorithm
            return nothing
        end
    end
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

    #Broadcast info to workers
    parallel_info = KirlikSayinParallelInfo(scalars, model, start_time, variables, algorithm)
    _distribute_info_to_workers!(parallel_info)

    # Ideal and Nadir point estimation
    results = Distributed.pmap(i -> begin  
        @info "Computing ideal and nadir point for objective $i"
        res = process_ideal_nadir_point(i, yI, yN, sharedIdealPoint)
        @info "Finished ideal and nadir point for objective $i"
        return res
    end, 1:n)
    
    model.subproblem_count += sum(r -> r.subproblems_added, results)

    # Check for errors during ideal/nadir point computation
    if any(r -> r.status != MOI.OPTIMAL, results)
        res = filter(r -> r.status != MOI.OPTIMAL, results)[1]
        # println("Error during ideal/nadir point computation. Status: ", res.status)
        return res.status, nothing
    end

    # updating the model info
    model.ideal_point = collect(sharedIdealPoint)
    
    L = [_RectangleParallel(_project(yI, k), _project(yN, k))]
    status = MOI.OPTIMAL
    while !isempty(L)
        if (ret = _check_premature_termination(model, start_time)) !== nothing
            status = ret
            break
        end

        volumes = [_volume(Rᵢ, _project(yI, k)) for Rᵢ in L]
        selected_volumes_idx = partialsortperm(volumes, 1:min(Distributed.nworkers(), length(L)), rev=true)
        selected_boxes = L[selected_volumes_idx]
        
        old_L_length = length(L)
        @info "Starting parallel box processing for $(length(selected_boxes)) boxes. Total boxes: $(length(L))"
        results = Distributed.pmap(box -> begin
            @info "\nStarted working on box: $box"
            res = process_box(box)
            @info "Finished working on box: $box"
            return res
        end, selected_boxes)
        @assert length(L) == old_L_length "Length of L changed during parallel processing!"

        # all_res = fetch.(futures)
        for res in results
            if res.should_remove
                _remove_RectangleParallel(L, _RectangleParallel(_project(yI, k), res.box.u))
            else
                if !(res.Y in YN)
                    push!(solutions, SolutionPoint(res.X, res.Y))
                    push!(YN, res.Y)
                    L = _update_list(L, res.Y_proj)
                end
                _remove_RectangleParallel(L, _RectangleParallel(res.Y_proj, res.box.u))
            end  

            # Update subproblem count
            model.subproblem_count += res.subproblems_increase   
        end
        
        # change status if no solutions found and L is empty
        if isempty(L) && length(solutions) == 0
            status = filter(r -> r.status != MOI.OPTIMAL, results)[1].status
        end
    end
    @info "Box processing complete. Number of solutions found: $(length(solutions))"
    return status, filter_nondominated(MOI.MIN_SENSE, solutions)
end


struct IdealNadirProcessResult
    status::MOI.TerminationStatusCode
    ideal_point::Float64
    nadir_point::Float64
    subproblems_added::Int64
end
function process_ideal_nadir_point(i::Int, yI::SharedArrays.SharedVector{Float64}, yN::SharedArrays.SharedVector{Float64}, sharedIdealPoint::SharedArrays.SharedVector{Float64})::IdealNadirProcessResult
    # Rebuild local model to avoid conflicts
    f_i = SCALARS[i]
    old_subproblem_count = LOCAL_MODEL.subproblem_count
    # Ideal point
    MOI.set(LOCAL_MODEL.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
    optimize_inner!(LOCAL_MODEL)
    status = MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return IdealNadirProcessResult(status, NaN, NaN, LOCAL_MODEL.subproblem_count - old_subproblem_count)
        # return status, nothing
    end
    _, Y = _compute_point(LOCAL_MODEL, VARIABLES, f_i)
    sharedIdealPoint[i] = yI[i] = Y
    
    # Nadir point
    MOI.set(LOCAL_MODEL.inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    optimize_inner!(LOCAL_MODEL)
    status = MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        # Repair ObjectiveSense before exiting
        MOI.set(LOCAL_MODEL.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        _warn_on_nonfinite_anti_ideal(ALGO, MOI.MIN_SENSE, i)
        return IdealNadirProcessResult(status, NaN, NaN, LOCAL_MODEL.subproblem_count - old_subproblem_count)
        # return status, nothing
    end

    _, Y = _compute_point(LOCAL_MODEL, VARIABLES, f_i)
    yN[i] = Y + δ
    MOI.set(LOCAL_MODEL.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return IdealNadirProcessResult(status, yI[i], yN[i], LOCAL_MODEL.subproblem_count - old_subproblem_count)
end

struct BoxProcessResult
    should_remove::Bool
    X::Union{Dict{MOI.VariableIndex,Float64}, Nothing}
    Y::Union{Vector{Float64}, Nothing}
    Y_proj::Union{Vector{Float64}, Nothing}
    box::_RectangleParallel
    subproblems_increase::Int64
    status::MOI.TerminationStatusCode
end

function process_box(box::_RectangleParallel)::BoxProcessResult
    uᵢ = box.u
    # Solving the first stage model: P_k(ε)
    #   minimize: f_1(x)
    #       s.t.: f_i(x) <= u_i - δ
    @assert K == 1
    MOI.set(
        LOCAL_MODEL.inner,
        MOI.ObjectiveFunction{typeof(SCALARS[K])}(),
        SCALARS[K],
    )
    ε_constraints = Any[]
    for (i, f_i) in enumerate(SCALARS)
        if i == K
            continue
        end
        ci = MOI.Utilities.normalize_and_add_constraint(
            LOCAL_MODEL.inner,
            f_i,
            MOI.LessThan{Float64}(uᵢ[i-1] - δ),
        )
        push!(ε_constraints, ci)
    end
    old_subproblem_count = LOCAL_MODEL.subproblem_count
    optimize_inner!(LOCAL_MODEL)
    if !_is_scalar_status_optimal(LOCAL_MODEL)
        # If this fails, it likely means that the solver experienced a
        # numerical error with this box. Just skip it.
        # _remove_RectangleParallel(L, _RectangleParallel(_project(yI, K), uᵢ))
        MOI.delete.(LOCAL_MODEL, ε_constraints)
        # println("First stage optimization failed with status: ", MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus()))
        return BoxProcessResult(true, nothing, nothing, nothing, box, LOCAL_MODEL.subproblem_count - old_subproblem_count, MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus()))
    end
    zₖ = MOI.get(LOCAL_MODEL.inner, MOI.ObjectiveValue())
    # Solving the second stage model: Q_k(ε, zₖ)
    # Set objective sum(model.f)
    sum_f = MOI.Utilities.operate(+, Float64, SCALARS...)
    MOI.set(LOCAL_MODEL.inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
    # Constraint to eliminate weak dominance
    zₖ_constraint = MOI.Utilities.normalize_and_add_constraint(
        LOCAL_MODEL.inner,
        SCALARS[K],
        MOI.EqualTo(zₖ),
    )
    optimize_inner!(LOCAL_MODEL)
    if !_is_scalar_status_optimal(LOCAL_MODEL)
        # If this fails, it likely means that the solver experienced a
        # numerical error with this box. Just skip it.
        MOI.delete.(LOCAL_MODEL, ε_constraints)
        MOI.delete(LOCAL_MODEL, zₖ_constraint)
        # _remove_RectangleParallel(L, _RectangleParallel(_project(yI, k), uᵢ))
        # println("Second stage optimization failed with status: ", MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus()))
        return BoxProcessResult(true, nothing, nothing, nothing, box, LOCAL_MODEL.subproblem_count - old_subproblem_count, MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus()))
    end
    X, Y = _compute_point(LOCAL_MODEL, VARIABLES, LOCAL_MODEL.f)
    Y_proj = _project(Y, K)
    #  _remove_RectangleParallel(L, _RectangleParallel(Y_proj, uᵢ))
    MOI.delete.(LOCAL_MODEL, ε_constraints)
    MOI.delete(LOCAL_MODEL, zₖ_constraint)
    return BoxProcessResult(false, X, Y, Y_proj, box, LOCAL_MODEL.subproblem_count - old_subproblem_count, MOI.get(LOCAL_MODEL.inner, MOI.TerminationStatus()))
end

function old_code()
    #region old code
        max_volume_index = argmax([_volume(Rᵢ, _project(yI, k)) for Rᵢ in L])
        uᵢ = L[max_volume_index].u
        # Solving the first stage model: P_k(ε)
        #   minimize: f_1(x)
        #       s.t.: f_i(x) <= u_i - δ
        @assert k == 1
        MOI.set(
            model.inner,
            MOI.ObjectiveFunction{typeof(scalars[k])}(),
            scalars[k],
        )
        ε_constraints = Any[]
        for (i, f_i) in enumerate(scalars)
            if i == k
                continue
            end
            ci = MOI.Utilities.normalize_and_add_constraint(
                model.inner,
                f_i,
                MOI.LessThan{Float64}(uᵢ[i-1] - δ),
            )
            push!(ε_constraints, ci)
        end
        optimize_inner!(model)
        if !_is_scalar_status_optimal(model)
            # If this fails, it likely means that the solver experienced a
            # numerical error with this box. Just skip it.
            _remove_RectangleParallel(L, _RectangleParallel(_project(yI, k), uᵢ))
            MOI.delete.(model, ε_constraints)
            # continue
            return
        end
        zₖ = MOI.get(model.inner, MOI.ObjectiveValue())
        # Solving the second stage model: Q_k(ε, zₖ)
        # Set objective sum(model.f)
        sum_f = MOI.Utilities.operate(+, Float64, scalars...)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
        # Constraint to eliminate weak dominance
        zₖ_constraint = MOI.Utilities.normalize_and_add_constraint(
            model.inner,
            scalars[k],
            MOI.EqualTo(zₖ),
        )
        optimize_inner!(model)
        if !_is_scalar_status_optimal(model)
            # If this fails, it likely means that the solver experienced a
            # numerical error with this box. Just skip it.
            MOI.delete.(model, ε_constraints)
            MOI.delete(model, zₖ_constraint)
            _remove_RectangleParallel(L, _RectangleParallel(_project(yI, k), uᵢ))
            # continue
            return
        end
        X, Y = _compute_point(model, variables, model.f)
        Y_proj = _project(Y, k)
        if !(Y in YN)
            push!(solutions, SolutionPoint(X, Y))
            push!(YN, Y)
            L = _update_list(L, Y_proj)
        end
        _remove_RectangleParallel(L, _RectangleParallel(Y_proj, uᵢ))
        MOI.delete.(model, ε_constraints)
        MOI.delete(model, zₖ_constraint)
        #endregion
end
