using BenchmarkTools
using DataFrames
include("approximateMethods.jl")
# Numerical Experiments.

# We run four experiments: The first two find the intersection between two ellipses, and the last two find the intersection between an ellipse and a half-plane.

# In each case, we consider two scenarios: the intersection has an empty interior or not.


R(θ) = [cos(θ) sin(θ);-sin(θ) cos(θ)]

# First, we define some sets:

# First ellipse. We will find the intersection between this set and 4 other sets.

A = diagm([2.0,1/5.0])*R(-π/4)
M = Symmetric(A'*A)
z0=[0.0,0]
ell = Ellipsoid(z0,M) 

# One ellipse that intersects with the first ellipse, with a nonempty interior.
B = diagm([1/2.0,5.0/2])*R(π/3)
M1 = Symmetric(B'*B)
z1 = [2.3,1.0/2]
ell2 = Ellipsoid(z1,inv(M1))

# Another ellipse that intersects with the first ellipse, with an empty interior (the intersection is just a point on the boundary of both ellipses).

d = [1.273;-1.1968]         # Pick a direction
x = d/sqrt(dot(inv(M)*d,d)) # Find a point on the boundary
dd = inv(M1)*inv(M)*x 
λ = dd'*M1*dd
z2 = dd/λ + x
M2 = M1*λ
ell3 = Ellipsoid(z2,inv(M2))

## Hyperplanes

hh = HalfSpace([-1.0,0],-1.3)  # Nonempty interior.
xm = linearOptimiser(ell)([1.0,0])[1]
h2 = HalfSpace([-1.0,0],-xm[1]) # Empty interior.

## Experiments
x0 = [-1,1.5] # The starting point for each of our experiments.

project(ell::Ellipsoid,x) = aproject(ell,x)

"""
Run an experiment over two sets, for selected parameters.

called using:
```julia
julia> experiment(x0,ell,ell2)
julia> experiment(x0,ell,ell3)
```

# Replace the projection with 

"""
experiment(x0,A,B;values = [0.0,0.120,0.245]) = Dict([ε => aDR(x0,A,B,ε) for ε ∈ values])

# We call our experiment as:
# > 
# > experiment(x0,ell,ell2)
# etc.

testvalues = [0,0.120,0.245]


suite = BenchmarkGroup()
for s2 in ["ellipse","half-space"]
  suite[s2] = BenchmarkGroup()
  for i in ["nonempty","empty"]
    suite[s2][i] = BenchmarkGroup(["ε","Algorithm"])
  end
end

for ε ∈ testvalues
  suite["ellipse"]["nonempty"][ε] = @benchmarkable aDR($x0,$ell,$ell2,$ε)
  suite["ellipse"]["empty"][ε] = @benchmarkable aDR($x0,$ell,$ell3,$ε)
  suite["half-space"]["nonempty"][ε] = @benchmarkable aDR($x0,$ell,$hh,$ε)
  suite["half-space"]["empty"][ε] = @benchmarkable aDR($x0,$ell,$h2,$ε)
end
suite["half-space"]["nonempty"][0.499] = @benchmarkable aDR($x0,$ell,$hh,0.499)
suite["half-space"]["empty"][0.499] = @benchmarkable aDR($x0,$ell,$h2,0.499)

tune!(suite)

results = run(suite)

outcomes = Dict(s2 => Dict(i => Dict(k => time(ratio(judge(mean(results[s2][i][k]),mean(results[s2][i][testvalues[1]]))))
             for k in keys(results[s2][i]))
            for i in ["nonempty","empty"])
            for s2 in ["ellipse","half-space"]
           )

outcomes = [(s2, i,k, time(ratio(judge(mean(results[s2][i][k]),mean(results[s2][i][testvalues[1]])))))
            for s2 in ["ellipse","half-space"]
            for i in ["nonempty","empty"]
             for k in keys(results[s2][i])
           ]

outcomes = rename(DataFrame(outcomes),[:set,:interior,:ϵ,:ratio])

## Some functions to output to files.

tikz(x) = "($(x[1]),$(x[2]))"
node(x,text="") = tikz(x) * " node {$text}"
path(p::Tuple{Vector{Float64},Vector{Float64}}) = tikz(p[1]) * " -- " * node(p[2])
nodes(v::Vector{Vector{Float64}}) = join(node.(v)," -- ") * ";"
paths(v::Vector{Tuple{Vector{Float64},Vector{Float64}}}) = join(path.(v),"\n") * ";"


