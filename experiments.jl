using BenchmarkTools
using DataFrames
#using RandomMatrices
include("approximateMethods.jl")
# Numerical Experiments.

import BenchmarkTools: tune!,run

struct Experiment
  sets    :: Vector{LazySet}
  suite   :: BenchmarkGroup
end

tune!(exp::Experiment) = tune!(exp.suite)
run(exp::Experiment) = run(exp.suite)

function viewresults(results::BenchmarkGroup)
  results = [vcat(collect(x[1]),x[2]) for x in leaves(results)]
  keys = [:set,:interior,:ϵ,:trial]
  df = DataFrame([Dict(zip(keys,r)) for r in results])
  df[!,:benchmark] .= df[!,:trial]
  df[!,:ratio] .= 0.0
  for x in eachrow(df)
    y = filter(z -> z[:ϵ] == 0.0 && all([z[k] == x[k] for k in [:set,:interior]]),df)
    x[:benchmark] = y[1,:trial]
  end
  df[!,:ratio] .= time.(ratio.(mean.(df[!,:trial]),mean.(df[!,:benchmark])))
  return df
end


function randomEllipsoid(diagon)
  dim = length(diagon)
  A = rand(dim,dim)
  Q, R = qr(A)
  M = Symmetric(Q'*diagm(diagon.^2)*Q)
  return Ellipsoid(zeros(dim),M)
end

"Find a point on the boundary of an ellipse"
function randomPoint(ell::Ellipsoid)
  M = shape_matrix(ell)
  d = rand(dim(ell)) - 0.5*ones(dim(ell))
  return d/sqrt(dot(inv(M)*d,d))
end

function findPt(ell::Ellipsoid,ell2::Ellipsoid,x,δ=0.)
  M = shape_matrix(ell)
  M1 = shape_matrix(ell2)
  gg  = M\x
  dd = M1*inv(M)*x 
  λ = sqrt(gg'*M1*gg)
  z2 = dd/λ + x + δ*dd
  return Ellipsoid(z2,M1)
end

function findHyperplane(ell::Ellipsoid,x,δ=0.)
  M = shape_matrix(ell)
  gg  = M\x
  gg = -normalize(gg)
  return HalfSpace(gg,gg'*x)
end

function Experiment(ndim :: Integer=2,testvalues = [0,0.125,0.25,0.49])
  
  x0 = ones(ndim)*2
  # The starting point for each of our experiments.
  # Step 1: Generate an ellipse with eigenvalues between 1/5 and 2, evenly distributed.
  ell = randomEllipsoid(range(0.2,2,ndim))
  # Step 2: Generate a hyperplane that intersects with the ellipse:
  ell2 = randomEllipsoid(range(5.0/2,1.0/2,ndim))

  # Step 3: Find a random point on the ellispoids, with the same gradient: 
  x = randomPoint(ell)
  ell3 =  findPt(ell,ell2,x)

  println("Ellipsoid: first experiment.")
  aDR(x0,ell,ell3,0.125)
  ell4 =  findPt(ell,ell2,x,-0.02)
  println("Ellipsoid: second experiment.")
  aDR(x0,ell,ell4,0.125)

  # Step 4: Find the vertical hyperplanes
  hh1 = findHyperplane(ell,x)
  hh2 = findHyperplane(ell,x,-0.2)

  println("Half-Space: first experiment.")
  aDR(x0,ell,hh1,0.125)
  println("Half-Space: second experiment.")
  aDR(x0,ell,hh2,0.125)

  sets = Dict(
     "ellipse" => Dict("empty" => ell3,"nonempty" => ell4),
     "half-space" => Dict("empty" => hh1,"nonempty" => hh2))

  # Construct the suite of experiments.
  suite = BenchmarkGroup()
  for s2 in ["ellipse","half-space"]
    suite[s2] = BenchmarkGroup()
    for i in ["nonempty","empty"]
        suite[s2][i] = BenchmarkGroup(["ε","Algorithm"])
  for ε ∈ testvalues
    suite[s2][i][ε] = @benchmarkable aDR($x0,$ell,$sets[$s2][$i],$ε)
    #suite["empty"][ε] = @benchmarkable aDR($x0,$ell,$ell3,$ε)
    #suite["empty"][ε] = @benchmarkable aDR($x0,$ell,$ell5,$ε)
#    suite["half-space"]["nonempty"][ε] = @benchmarkable aDR($x0,$ell,$hh,$ε)
#    suite["half-space"]["empty"][ε] = @benchmarkable aDR($x0,$ell,$h2,$ε)
  end
end end
  return Experiment([ell,ell3,ell4,hh1,hh2],suite)
#  suite["half-space"]["nonempty"][0.499] = @benchmarkable aDR($x0,$ell,$hh,0.499)
#  suite["half-space"]["empty"][0.499] = @benchmarkable aDR($x0,$ell,$h2,0.499)

end


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

# tune!(suite)
# 
# results = run(suite)
# 
# outcomes = Dict(s2 => Dict(i => Dict(k => time(ratio(judge(mean(results[s2][i][k]),mean(results[s2][i][testvalues[1]]))))
#              for k in keys(results[s2][i]))
#             for i in ["nonempty","empty"])
#             for s2 in ["ellipse","half-space"]
#            )
# 
# outcomes = [(s2, i,k, time(ratio(judge(mean(results[s2][i][k]),mean(results[s2][i][testvalues[1]])))))
#             for s2 in ["ellipse","half-space"]
#             for i in ["nonempty","empty"]
#              for k in keys(results[s2][i])
#            ]
# 
# outcomes = rename(DataFrame(outcomes),[:set,:interior,:ϵ,:ratio])
# 


## Some functions to output to files.

tikz(x) = "($(x[1]),$(x[2]))"
node(x,text="") = tikz(x) * " node {$text}"
path(p::Tuple{Vector{Float64},Vector{Float64}}) = tikz(p[1]) * " -- " * node(p[2])
nodes(v::Vector{Vector{Float64}}) = join(node.(v)," -- ") * ";"
paths(v::Vector{Tuple{Vector{Float64},Vector{Float64}}}) = join(path.(v),"\n") * ";"


