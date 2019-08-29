using Random 
using LinearAlgebra
using Statistics
using Plots 

function sigmoid(X)

    n ,m = size(X)
    sigma = zeros(n,m)
    for i=1:n
        for j=1:m

            sigma[i,j] = 1/(1 + exp(-X[i,j]))
            
        end
    end 

    return sigma , X

end


function relu(X)

    n , m = size(X)
    rel = zeros(n,m)
    for i=1:n
        for j=1:m

            rel[i,j] = max( 0 , X[i , j])
            
        end
    end 

    return rel , X

end


function init_param(layer_dimensions)

    param = Dict()
    
    for l=2:length(layer_dimensions)

        param[string("W_" , string(l-1))] = rand(layer_dimensions[l] , layer_dimensions[l-1])*0.1
        param[string("b_" , string(l-1))] = zeros(layer_dimensions[l] , 1)
    
    end

    return param

end


function forward_linear(A , w , b)

    Z = w*A 
    Z = Z .+ b
    cache = (A , w , b)

    return Z,cache

end

function cost_function(AL , Y)
    
    cost = -mean(Y.*log.(AL) + (1 .- Y).*log.(1 .- AL))

    return cost 

end

function backward_linear_step(dZ , cache)

    A_prev , W , b = cache

    m = size(A_prev)[2]
    dW = dZ * (A_prev')/m
    db = sum(dZ , dims = 2)/m
    dA_prev = (W')* dZ
   # println("dW ->" , dW)
   # println("dA_prev ->" , db)
   # println("dA_prev" , dA_prev)
    return dW , db , dA_prev 

end

function backward_relu(dA , cache_activation)
    return dA.*(cache_activation.>0)

end 

function backward_sigmoid(dA , cache_activation)
    return dA.*(sigmoid(cache_activation)[1].*(1 .- sigmoid(cache_activation)[1]))
end

function backward_activation_step(dA , cache , activation)

    linear_cache , cache_activation = cache
    if (activation == "relu")

        dZ = backward_relu(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)

    elseif (activation == "sigmoid")

        dZ = backward_sigmoid(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)

    end 

    return dW , db , dA_prev

end 

function model_backwards_step(A_l , Y , caches)

    grads = Dict()

    L = length(caches)

    m = size(A_l)[2]

    Y = reshape(Y , size(A_l))
    dA_l = (-(Y./A_l) .+ ((1 .- Y)./( 1 .- A_l)))
  #  println("dA_l" , dA_l)
    current_cache = caches[L]
    grads[string("dW_" , string(L))] , grads[string("db_" , string(L))] , grads[string("dA_" , string(L-1))] = backward_activation_step(dA_l , current_cache , "sigmoid")
    for l=reverse(1:L-1)

        current_cache = caches[l]
        grads[string("dW_" , string(l))] , grads[string("db_" , string(l))] , grads[string("dA_" , string(l-1))] = backward_activation_step(grads[string("dA_" , string(l))] , current_cache , "relu")
    end 

    return grads 

end

function update_param(parameters , grads , learning_rate)

    L = Int(length(parameters)/2)

    for l=1:(L)

        parameters[string("W_" , string(l))] -= learning_rate.*grads[string("dW_" , string(l))]
        parameters[string("b_",string(l))] -= learning_rate.*grads[string("db_",string(l))]

    end 

    return parameters

end

function calculate_activation_forward(A_pre , W , b , function_type)

    if (function_type == "sigmoid")

        Z , linear_step_cache = forward_linear(A_pre , W , b)
      #  println("Z" , Z)
        A , activation_step_cache = sigmoid(Z)

    elseif (function_type == "relu")

        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = relu(Z)

    end

    cache = (linear_step_cache , activation_step_cache)
    return A , cache

end

function model_forward_step(X , params)

    all_caches = []
    A = X
    L = length(params)/2

    for l=2:L
        A_pre = A
        A , cache = calculate_activation_forward(A_pre , params[string("W_" , string(Int(l-1)))] , params[string("b_" , string(Int(l-1)))] , "relu")
        push!(all_caches , cache)
    end 
    A_l , cache = calculate_activation_forward(A , params[string("W_" , string(Int(L)))] , params[string("b_" , string(Int(L)))] , "sigmoid")
    push!(all_caches , cache)


    return A_l , all_caches 

end


function train_nn(layers_dimensions , X , Y , learning_rate , n_iter)

    params = init_param(layers_dimensions)
    costs = []
    iters = []
   # anim = @animate for i=1:n_iter
    for i=1:n_iter
        println("Params ->" , params["W_4"])
     #   plt = plot(1, title = "Cost function vs N_Iter")
        A_l , caches  = model_forward_step(X , params)
      #  println("A_l" , A_l)
        cost = cost_function(A_l , Y)
        
        grads  = model_backwards_step(A_l , Y , caches)
        println("Grads ->" , grads["dW_4"])
        params = update_param(params , grads , learning_rate)
        println("Iteration ->" , i)
        println("Cost ->" , cost)
       # println("Iters -> " , iters)
        push!(iters , i)
        push!(costs , cost)
      #  plot!(iters, costs )
        
    end 
    plt = plot(iters , costs))
    gui(plt)
    return params , costs 

end

X = rand(10,1000)
Y = rand([0,1] , 1000)

layers_dimensions = (10 , 50 ,5, 3, 1)

param , cost = train_nn(layers_dimensions , X , Y , 0.05 , 1000)
