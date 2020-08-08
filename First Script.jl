using DelimitedFiles
using Random

# Turns CSV into an Array/Matrix object
all_data = readdlm("C:\\Users\\chris_000\\Pictures\\Regression\\Personal\\airfoil_self_noise.tsv", '\t', '\n')

function main(full_data)
    model , SSE = modelRepeater(full_data)
    return model, SSE
end

function LinearRegression(full_data)
    dummy_ones = [1 for n in 1:length(full_data[:,1])]                          #The ones are used in the matrix for the y intercept
    #Y = full_data[:,length(full_data[1,:])]                                    #The thing we try to predict (Y) MUST be the last column
    #full_data = full_data[1:end, 1:end .!= length(full_data[1,:])]             #Deletes the last column because that's now Y
    Y, full_data = separateColumns(full_data)
    X = hcat(dummy_ones, full_data)
    X_t = transpose(X)
    coeff_est = inv(X_t*X)*X_t*Y                                                #Formula to get regression coefficients
    return coeff_est
end

function randChoiceIndex(n)
    index_array = []
    while length(index_array) < ceil(0.35*n)                                    #Uses about 35% of our data for testing, can change
        i = rand(1:n)
        if !(i in index_array)
            append!(index_array, i)
        end
    end
    return index_array                                                          #Random unique numbers in an array in given range n
end

function bootstrap(full_data)
    index_array = randChoiceIndex(length(full_data[:,1]))
    test_set = reshape(full_data[index_array[1],:], 1, length(full_data[1,:]))  #Start making a test set using the first random data point
    popfirst!(index_array)                                                      #Delete the first data point to avoid redundance with last line
    for n in index_array
        new_line = reshape(full_data[n,:], 1, length(full_data[1,:]))           #Had to use reshape() to make the result a matrix
        test_set = vcat(test_set, new_line)                                     #Appending a data point to the end of the test matrix
    end
    coefficient_estimates = LinearRegression(test_set)
    return coefficient_estimates
end

function findSSE(model, full_data)
    Y, full_data = separateColumns(full_data)
    full_data = hcat(ones(length(full_data[:,1]), 1), full_data)                #Makes column of 1's for intercept
    SSE = float((Y[1,:] .- sum(model .* full_data[1,:]))[1])^2                  #SSE stands for sum of square errors
    for i in 2:length(full_data[:,1])
        SSE += float((Y[i,:] .- sum(model .* full_data[i,:]))[1])^2
    end
    return(SSE)                                                                 #SSE is a basic measure for model strength: we want it small
end

function modelRepeater(full_data)
    model = bootstrap(full_data)
    SSE = findSSE(model, full_data)
    for i in 1:100                                                              #Creates a model 100 times trying to find one with small error
        temp_model = bootstrap(full_data)
        temp_SSE = findSSE(temp_model, full_data)
        if temp_SSE < SSE
            model = temp_model
            SSE = temp_SSE
        end
    end
    return (model, SSE)                                                         #Returns a tuple of the model (array) and SSE (see findSSE())
end

function separateColumns(full_data)
    Y = full_data[:,length(full_data[1,:])]                                     #Makes a vector of correct answers
    full_data = full_data[1:end, 1:end .!= length(full_data[1,:])]              #Deletes redundant Y column from data
    return (Y, full_data)
end

temp, SSE = main(all_data)
println(temp)
println(SSE)
