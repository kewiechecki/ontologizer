using CSV: normalizename
#using SparseArrays: cholmod_sparse_struct
#using Leiden: refine_partition
include("ontology.jl")

using CSV, DataFrames, Tensors, CategoricalArrays

γs = rand(Uniform(0.1,1),100)

dat = DataFrame(CSV.File("z_dat.csv",normalizenames=true))
X = Matrix(dat[:,2:end])
X = scaledat(X)
X = hcat(filter(x->sum(x) != 0,eachslice(X,dims=2))...)
test,train = sampledat(X',10)

Ω = newontology(whichencoder,(t,E,f)->clustblock(t,E,γs,1:100,f),train)
Ω = nsteps(5,:refine)(Ω)
Ω = nsteps(10,:refine)(Ω)

J = predict(Ω,test)

map(n->Flux.mse(test,foldr(+,J[1][1:n])),1:length(Ω.bfTheta))
map(Θ-> maximum(Θ.clusters),Ω.bfTheta)
map(maximum,J[2])

clustfn = Ω.clusterfn
ks=3:100
    m = whichencoder(train|>gpu)
    E = m.encoder(train|>gpu)|>cpu
    tree = KDTree(E)
    G,clusts,hyperparams = clustfn(tree,E)
    println(hyperparams)

    Xhat = m.decoder(Ehat|>gpu)|>cpu
    clusts = clustcol(clusts.partition)
clusts = hcat(map(T->T.clusters,Ω.bfTheta)...)
Y = Ω.bfTheta[1].theta.encoder(train|>gpu)|>cpu
tmp = clust(Ω.bfTheta[1].G,Ω.bfTheta[1].hyperparameters[2])
E_J = map(i->Y[:,i],tmp.partition)
M = map(x->mean(x,dims=2),E_J)
W = map(zca,E_J)
E_W = map(*,W,E_J)

Ehat = map(tmp.partition,E_W,M) do i, E, mu
    Ehat = zeros(size(Y))
    Ehat[:,i] .=  E .+ mu
    return Ehat
end

Ehat = foldr(+,Ehat)

Yhat = Ω.bfTheta[1].theta.decoder(Ehat|>gpu)|>cpu

Y = train[:,tmp.partition[1]]
    mu = mean(Y,dims=2)
    Y = Y .- mu
    Sigma = cov(Y,dims=2)
    Lambda,U = eigen(Sigma)
    Lambdaneghalf = 1 ./ sqrt.(Lambda)
    W = U * Diagonal(Lambdaneghalf) * U'

using InformationMeasures
function f(clusts)
    mi = mapslices(x->mapslices(y->get_mutual_information(x,y),
                                clusts',dims=2),
                   clusts,dims=1)
    h = mapslices(get_entropy,clusts,dims=1)

    return mi .*2 ./ (h.+h')
end

using Plots
heatmap(log.(f(clusts(Ω))))
heatmap(f(clusts'))
smi = mapslices(x->mapslices(y->get_mutual_information(x,y),clusts,dims=2),clusts',dims=1)
ci = mapslices(x->mapslices(y->get_cross_entropy(x,y),clusts',dims=2),clusts,dims=1)

groups = CSV.read("groups.csv",DataFrame,normalizenames=true,
                  select=["Condition","Phenotype","orientation_perturbed",
                          "migration_perturbed","division_perturbed"])
groups = map(categorical,eachcol(groups))

X̂ = predict(Ω,X)
clusts = hcat(X̂[2]...)

using Clustering, Distances, StatsPlots
condmi = mapslices(x->map(y->mutualinfo(x,levelcode.(y),normed=false),groups),clusts,dims=1)
heatmap(condmi')
mapslices(x->mapslices(y->mutualinfo(x,y),clusts,dims=2),clusts',dims=1)
group_matrix = Matrix(groups[:,["Condition", "Phenotype"]])

result = mapslices(x -> mapslices(y -> mutualinfo(x, Int.(y)), group_matrix, dims=2), clusts, dims=1)


clusts = map(gamma->clust(Ω.bfTheta[2].G,gamma),γs)
Js = map(clustmat,clusts)

mi = map(c->get_mutual_information(Ω.bfTheta[1].clusters,clustcol(c.partition)),clusts)
argmin(mi)

function hc(metric,X,dims)
    return hclust(pairwise(metric,X,dims=dims),
                    branchorder=:optimal)
end

function hm(X,metric=Euclidean())
    cols = hc(metric,X,2)
    rows = hc(metric,X,1)
    heatmap(X[cols.order,rows.order])
end

hm(f(clusts))

# Dummy data for illustration
data = f(clusts')

# Euclidean distance
distances = pairwise(Euclidean(), data, dims=2)

# Hierarchical clustering
result = hclust(distances)

# Ordered data based on the clustering result
ordered_data = data[result.order, hclust(pairwise(Euclidean(),data,dims=1)).order]

# Create heatmap with clustering
heatmap(ordered_data, color=:viridis)
