using Flux, ProgressMeter, NearestNeighbors, InvertedIndices,
    TensorOperations, TensorCast, Statistics, Leiden,
    Distributions, LinearAlgebra, SparseArrays, CUDA,
    Combinatorics, Distances
#include("generics.jl")

mutable struct Autoencoder
    encoder::Chain #Observation -> Embedding
    decoder::Chain #Embedding -> Observation
end

mutable struct Typology
    X::AbstractArray{Union{Float32,Float64}}
    theta::Autoencoder
    tree::KDTree
    D::AbstractMatrix
    G::SparseMatrixCSC
    clusters#::AbstractArray{Int,1}
    M#::AbstractArray{Union{Float32,Float64}}
    W#::AbstractArray{AbstractMatrix{Union{Float32,Float64}}}
    Ehat::AbstractArray{Union{Float32,Float64}}
    Xhat::AbstractArray{Union{Float32,Float64}}
    hyperparameters::Tuple
end

mutable struct Ontology
    encoderfn::Function
    clusterfn::Function
    bfTheta::AbstractArray{Typology,1}
    R::AbstractArray{Union{Float32,Float64}}
end

function newontology(encoderfn,decoderfn,X)
    #(Matrix -> Autoencoder) -> (Matrix -> Typology) -> Matrix -> Ontology
    Ontology(encoderfn,decoderfn,[],X)
end

function refine(Omega::Ontology)
    #Ontology -> Ontology
    Theta = typology(Omega.encoderfn,Omega.clusterfn,Omega.R)
    Omega.bfTheta = cat(Omega.bfTheta,Theta,dims=1)
    Omega.R = Omega.R - Theta.Xhat
    return Omega
end

function predict(Theta::Typology,X::AbstractArray)
    #Typology -> Matrix -> Matrix
    #E = Theta.theta.encoder(X|>gpu)|>cpu
    E = encode(Theta,X)
    g,d = knn(Theta.tree,E,Theta.hyperparameters[1])
    clusts = map(n->mode(clusts(Theta)[n]),g)
    Ehat = map(eachslice(E),clusts) do e,c
        ehat = (Theta.W[c] * (e .- Theta.M[c])) .+ Theta.M[c]
        return ehat
    end
        
    #Ehat = Theta.Ehat[:,clusts]
    #Xhat = Theta.theta.decoder(Ehat|>gpu)|>cpu
    Xhat = decode(Theta,Ehat)
    return clusts,Xhat
end

function predict(Omega::Ontology, X::AbstractArray)
    init = (Vector{AbstractMatrix}(), Vector{AbstractVector}(), X)

    result = foldl(Omega.bfTheta, init=init) do acc, Theta
        Xs, Cs, R = acc
        C, X = predict(Theta, R)
        push!(Xs, X)
        push!(Cs, C)
        R -= X
        return (Xs, Cs, R)
    end

    return result[1], result[2]
end

function encode(Theta::Typology,X::AbstractArray)
    return Theta.theta.encoder(X|>gpu)|>cpu
end

function encode(Omega::Ontology,X::AbstractArray,layer=1)
    return encode(Omega.bfTheta[layer],X)
end

function decode(Theta::Typology,E::AbstractArray)
    return Theta.theta.decoder(E|>gpu)|>cpu
end

function decode(Omega::Ontology,E::AbstractArray,layer=1)
    return decode(Omega.bfTheta[layer],E)
end

function clusts(Theta::Typology)
    return clustcol(Theta.clusters.partition)
end
    
function clusts(Omega::Ontology)
    return hcat(map(clusts,Omega.bfTheta)...)
end

#function loss(E,K)
#    #Matrix -> Matrix -> Float
#    return Flux.mse(E,matmul(K,E))
#end

function aic(nvars,loss)
    #Int -> [Float] -> [Float]
    return 2 .* nvars .- 2 .* log2.(1 .- loss)
end

#function matmul(K::AbstractMatrix,X::AbstractMatrix)
#/    #Matrix -> Matrix -> Matrix
#    K = Matrix(K)
#    X = Matrix(X)
#    @tensor E[k,j] := K[i,j] * X[k,i]
#    return E
#end
#
#function colsum(X::AbstractMatrix)
#    #Matrix -> Matrix
#    @reduce W[j] := sum(i) X[i,j]
#    return W
#end
#
#function rowsum(X::AbstractMatrix)
#    #Matrix -> Matrix
#    @reduce W[i] := sum(j) X[i,j]
#    return W
#end

function scaledat(X::AbstractArray,dims=1)
    #Y = mapslices(maximum,abs.(X),dims=dims)[1,:]
    #X = eachslice(X,dims=2)./Y
    #X = X[Y.!=0]
    #X = transpose(hcat(X...))
    #return Matrix(X)
    Y = X ./ maximum(abs.(X),dims=dims)
    Y[isnan.(Y)] .= 0
    return Y
end

function sampledat(X::AbstractArray,frac=10)
    j = size(X)[2]
    sel = sample(1:j,j ÷ frac)
    test = X[:,sel]
    train = X[:,Not(sel)]
    return test,train
end

function encoder(X::AbstractArray, n::Int, epochs=1000, batchsize=1024, wd=0.1, eta=0.01)
    i = size(X)[1]
    model = Chain(
        Dense(i => n, relu),
        Dense(n => i)
    ) |> gpu

    opt = Flux.Optimiser(Flux.AdamW(eta), Flux.WeightDecay(wd))

    data_loader = Flux.DataLoader((X, X) |> gpu, batchsize=batchsize, shuffle=true)
    
    function loss(x, y)
        yhat = model(x)
        L = Flux.mse(yhat, y)
        # Apply custom weight decay or other modifications here
        return L
    end

    @showprogress for _ in 1:epochs
        Flux.train!(loss, Flux.params(model), data_loader, opt)
    end
    
    return Autoencoder(Chain(model[1]),Chain(model[2]))
end

function whichencoder(X::AbstractArray, n=1, epochs=1000, batchsize=1024, wd=0, eta=0.01)
    println("n = ",n)
    theta = encoder(X,n,epochs,batchsize,wd,eta)
    y = sum(sum(theta.encoder(X),dims=1).==0)
    println("zero embeddings = ",y)
    if(y<2)
        return theta
    else
        return whichencoder(X,n+1,epochs,batchsize,wd,eta)
    end
end

function partitionmat(J,len)
    #filter!(x->length(x)>1,J)
    #G = hcat(map(x->hcat(Combinatorics.permutations(x,2)...),J)...)
    #return sparse(G[1,:],G[2,:],1,len,len)
    G = sum(cat(map(J) do x
                    V = sparsevec(x,1,len)
                    return V * V'
                end...,dims=3),dims=3)
    return G[:,:,1]
end

function knnmat(tree,E,k,self=false)
    #Tree -> Matrix -> Int -> KNN
    m = length(tree.data)
    g,d = knn(tree,E, k)

    function filt(i,j)
        sel = j .== i
        if(any(sel))
            return j[Not(sel)]
        else
            return j[1:length(j)-1]
        end
    end
    
    if(self)
        g = map(filt,1:m,g)
        k = k-1
    end

    I = vcat(map(x->repeat([x],k),1:m)...)
    J = vcat(g...)
    G = sparse(I,J,1,m,m)
    return G
end

function wak(G)
    #Matrix -> Matrix
    #returns G with row sums normalized to 1
    #W = colsum(G)
    W = sum(G,dims=2)
    K = G ./ W
    K[isnan.(K)] .= 0
    return K
end

function maskid(G)
    #Matrix -> Matrix
    #Returns G with the diagonal set to 0
    i,j = size(G)
    return G .* ((Matrix(I,i,j) .- 1) .* (-1))
end

function whichloss(E,Ks)
    #L = map(K->loss(E,K),Ks)
    L = map(K->Flux.mse(E,(K*E')'),Ks)
    return argmin(L)
end

function modelloss(m,X,E,Ks)
    Ks = Ks |> gpu
    L = map(Ks|>gpu) do K
        println(typeof(K))
        Ehat = (K * E')'
        println(typeof(Ehat))
        return Flux.mse(m.decoder(Ehat),X)
    end
    return argmin(L)
end

function noise2self(lossfn,Gs)
    #Tensor -> [Matrix] -> Int
    #returns the index of G ∈ Gs that is the best predictor of E
    Gs = map(maskid,Gs)
    sizeok = all(map(G->size(G)[1]==size(G)[2],Gs))
    println("size OK: ", sizeok)
    Ks = map(wak,Gs)
    #L = map(K->loss(E,K),Ks)
    #return argmin(L)
    return lossfn(Ks)
end


function clust(K,res)
    #Matrix -> Float -> Matrix
    return Leiden.leiden((K.+K') .> 0,resolution=res)
end

function clustmat(H)
    #Leiden -> Matrix
    n = maximum(vcat(H.partition...))
    M = zeros(n,n)
    for P in H.partition
        for i in P
            M[i,P] .= 1
        end
    end
    return M
end

#function centroid(E,clusts)
#    Ehat = map(P -> mean(E[:, P], dims=2), clusts.partition)
#    Ehat = cat(Ehat..., dims=2)  # concatenate the results along the 2nd dimension
#    #Êₘ = transpose(Êₘ)  # transpose to match the expected dimensions
#    return Ehat
#end
#    
#function centroidmap(H,Xhat,dims)
#    M = zeros(dims...)
#    for i in zip(H.partition,1:size(Xhat)[2])
#        M[:,i[1]] .= Xhat[:,i[2]]
#    end
#    return M
#end


function clustblock(tree,E,γs,ks,lossfn) #(G->whichloss(E,G)))
    #Tree -> Matrix -> [Float] -> [Int] -> (Graph,Cluster,Hyperparameters)
    n,m = size(E)
    D = 1 ./ (pairwise(Euclidean(),E,E) + (I*Inf))

    Gs = map(k->knnmat(tree,E,k,true),ks.+1)
    whichk = noise2self(lossfn,map(G->G .* D,Gs))
    k = ks[whichk]
    G = Gs[whichk]

    println("Size of E: ", size(E))
    println("Size of G ", size(G))

    Cs = map(γ->Leiden.leiden((G + G') .* D,resolution=γ),γs)
    Ps = map(C->D .* partitionmat(C.partition,m),Cs)
    whichγ = noise2self(lossfn,Ps)
    γ = γs[whichγ]
    hyperparams = (k,γ)
    return D,Gs[whichk],Cs[whichγ],hyperparams
end


#function residual(m,X,E,clusts)
#    #Autoencoder -> Matrix -> Matrix -> Cluster -> Matrix
##    Ê = matmul(wak(maskid(Js[whichγ])),E)
#    #Êₘ = map(P->mean(E[P]),clusts.partition)
#    #Êₘ = reshape(Êₘ, :, 1)  # reshape to 2D matrix
#    Xhat = m.decoder(E|>gpu)|>cpu
#    Xhat = centroidmap(clusts,Xhat,size(X))
#    return Xhat
#end

function clustcol(clusts)
    I = vcat(clusts...)
    V = vcat(map(i->repeat([i],length(clusts[i])),1:length(clusts))...)
    return Vector(sparsevec(I,V))
end


function typology(encoderfn,clustfn,X)
    #(Data -> Autoencoder) -> (Tree -> Embedding -> (Partition,Hyperparameters)) -> Typology
    X = X |> gpu
    m = encoderfn(X)

    E = m.encoder(X|>gpu)
    lossfn = Gs->modelloss(m,X,E|>gpu,Gs)
    E = E |> cpu
        
    tree = KDTree(E)
    D,G,clusts,hyperparams = clustfn(tree,E,lossfn)
    println(hyperparams)

    E_J = map(i->E[:,i],clusts.partition)
    M = map(x->mean(x,dims=2),E_J)
    W = map(zca,E_J)
    E_W = map(*,W,E_J)
    #Ehat = centroid(E,clusts)
    Ehat = map(clusts.partition,E_W,M) do i, x, mu
        Ehat = zeros(size(E))
        Ehat[:,i] .=  x .+ mu
        return Ehat
    end
    Ehat = foldr(+,Ehat)
    println("size of Ê: ",size(Ehat))
    println("clusts: ",length(clusts.partition))
    #Xhat = residual(m,X,Ehat,clusts)
    #clusts = clustcol(clusts.partition)
    Xhat = m.decoder(Ehat|>gpu)|>cpu
    return Typology(X,m,tree,D,G,clusts,M,W,Ehat,Xhat,hyperparams)
end

function nsteps(n,expr)
    return eval(reduce((x,y) -> Expr(:call,:∘,x,y),repeat([expr],n)))
end

function anyapprox(X)
    X = sort(X)
    return any(map(i->isapprox(X[i],X[i+1]),1:length(X)-1)...)
end

function zca(X,dims=1)
    i,j = size(X)
    if j<2
        return zeros(i,i)
    end
    mu = mean(X,dims=2)
    X = X .- mu
    Sigma = cov(X,dims=2)
    Lambda,U = eigen(Sigma)
    #Lambda[Lambda .< 0] .= 0
    #Lambdaneghalf = 1 ./ sqrt.(Lambda .+ eps())
    Lambdaneghalf = 1 ./ sqrt.(abs.(Lambda))
    Lambdaneghalf = Lambdaneghalf .* ((Lambda .> 0) .- (Lambda .< 0))
    #Lambdaneghalf[abs.(Lambdaneghalf) .== Inf] .= 0
    W = U * Diagonal(Lambdaneghalf) * U'
    W = scaledat(W,dims)
    return W
end

function fitclust(X)
    μ = mean(X,dims=1)
    X = X .- μ
    Σ = 1 ./ cov(X,dims=1)
    Σ = Σ ./ sum(Σ,dims=1)
    Y = X ./ mapslices(norm,X,dims=2) * Σ
    return Y,μ
end
