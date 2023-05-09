using LazySets
using LinearAlgebra

import LazySets: Ellipsoid, center, an_element, shape_matrix, LazySet, HalfSpace,σ

## PART 1: Conditional Gradient.

# The approximate alternating projection methods is based on calling
# the conditional gradient method (Frank-Wolfe method) at each step
# to find an approximate projection to a point onto a set, so that 
# the approximate projection is *inside* the set.

@doc raw"""
The Conditional Gradient method (also called Frank Wolfe method) is an iterative method for solving problems of the form:
```math
\text{minimize } f(x)\text{ subject to } x\in X
```
where the function ``f`` is differentiable, and the set ``X`` is convex and compact.

By default the method uses a finite difference approximation of the gradient, and the default step size at iteration ``i`` is ``\frac{1}{i}``.

To use this method, initialise it as follows:
```
julia > m = ConditionalGradient(f,X,x₀)

julia > m = ConditionalGradient(f,X,x₀)
```
"""
struct ConditionalGradient{T,F1,F2,F3,F4}
  f  :: F1          # The objective function.
  x₀ :: Vector{T}   # The initial solution.
  LO :: F2          # The Linear Optimisation Oracle used to solve the linear subproblem.
  ∇f :: F3          # The gradient of the function $f$.
  lineSearch :: F4  # Line Search.
end

"""
Compute an approximation of the derivative via the finite difference method.
"""
finiteDifference(f;ε=1e-10) = x -> (f(x+ε)-f(x-ε))/2ε

ConditionalGradient(f,x₀,LO::Function) = ConditionalGradient(f,x₀,LO,finiteDifference(f), (x,s,i) -> 2/i)
ConditionalGradient(f,x₀,X) = ConditionalGradient(f,x₀,LinearOptimiser(X),finiteDifference(f), (x,s,i) -> 2/i)

function Base.iterate(m::ConditionalGradient)
  g = -m.∇f(m.x₀)             # First step: get the gradient of $f$.
  s = m.LO(g)              # Second step: solve a LO using the LO oracle.
  return (m.x₀,(m.x₀,g,s,1))
end

function Base.iterate(m::ConditionalGradient,(x,g,s,i)) 
  λ = m.lineSearch(x,s,i)        # Third step: perform a line search.
  x1 = x + λ*(s-x)               # Compute the next iterate.
  g1 = -m.∇f(x1)                 # First step: get the gradient of $f$.
  s1 = m.LO(g1)                  # Second step: solve a LO using the LO oracle.
  return (x1,(x1,g1,s1,i+1))     # Return the next iterate.
end

## Part 2: Approximate Projection onto an ellipsoid.


@doc raw"""
Generates a linear optimiser oracle for a convex set ``X``. The oracle solves the problem

```math
\text{minimise } \langle s,x\rangle, \text{ subject to } x\in X
```

This oracle will be used as part of the Conditional Gradient for minimising a convex function (such as the distance) in an ellipsoid.
"""
function linearOptimiser(ell::Ellipsoid)
  return x -> σ(x,ell)
  M = shape_matrix(ell)
  c = center(ell)
  function LO(x)
    p = M*x             # Find the direction
    d = p/sqrt((M*p)⋅p) # Normalise
    return c + d
  end
  return LO
end

"""
The Conditional Gradient algorithm for approximately projecting a point onto a set.
"""
struct ApproximateCG{P <: Function, T <: Real, S <: LazySet}
  y :: Vector{T} 
  s :: S
  x :: Vector{T}
  pred :: P
end

function Base.iterate(m::ApproximateCG) 
  LO = linearOptimiser(m.s)
  ls(x) = 0.5*(x-m.y)⋅(x-m.y) # Least Squares Function
  ∇ls(x) = (x-m.y)          # Gradient of the least squares function
  search(x,s,i) = min(1.0,((m.y-x)⋅(s-x)/((s-x)⋅(s-x))))
  cg = ConditionalGradient(ls,m.x,LO,∇ls,search)
  (x1,s1) = Base.iterate(cg)
  return (x1,(s1,cg))
end

function Base.iterate(m::ApproximateCG,((x,g,s,i),cg))
  #sl = g⋅(s-x)
  #if sl < φ(m.γ,m.θ,m.λ,m.x,m.y,x) return nothing end
  !m.pred(x,g,s,i)  || return nothing
  (x1,s1) = Base.iterate(cg,(x,g,s,i))
  return (x1,(s1,cg))
end

function stopping(ε,x,g,s,i)
  sl = g⋅(s-x)
  return sl < ε
end

"""
Approximate projection using the Conditional Gradient method.
"""
function aproject(s,y;x = an_element(s), pred = (x,g,s,i) -> stopping(1e-6,x,g,s,i))
  if(y in s) return y end
  acg = ApproximateCG(y,s,x,pred)
  yp = Float64[]
  for outer yp in acg end;
  return yp
end

# For projecting onto a hyperplane, we just compute the exact projection, since it is straightforward.
aproject(s::HalfSpace,y;x = an_element(s), pred = s -> stopping(1e-6,s...)) = project(s,y)

## Exact Projection onto an Ellipsoid

# The exact projection $X$ onto an ellipsoid centered at c must satisfy the equation $λM(y-x) = c-x$ and $x^TMx = 1$, for some $λ$. We first find $λ$, then $x$. We use the following procedure:
# 1. Decompose the matrix $M=Q^TDQ$.
# 2. Set $v = Qx$, $u=Qc$ and $z=Qy+Du$. The equation to solve becomes: $λD(v-u) = Qy-v$ and $v^TDv = 1$.
# 3. Therefore, $v = (λD+I)^{-1}z$.
# 4. Since $v^TDv = 1$, then $λ$ must satisfy the equation: $z^T(λD+I)^{-1}D(λD+I)^{-1}z =⟨D(λD+I)^{-1}(z+Du),(λD+I)^{-1}(z+Du)⟩ = ∥(λD+I)^{-1}z∥_D^2  = 1$.
# 5. We apply Newton's method for solving this equation. Then, after finding $λ$, we find $v$.

raw"$∥(aD+I)^{-1}z∥_D^2  - 1$"
g(dd,z,a) = sum([d*v^2/(1+a*d)^2 for (d,v) in zip(dd,z)])-1;

raw"$∇(∥(aD+I)^{-1}z∥_D^2  - 1)$"
function dg(dd,z,a)
  d2 = sum([d*v^2/(1+d*a)^2 for (d,v) in zip(dd,z)]) - 1
  d3 = sum([2*d^2*v^2/(1+d*a)^3 for (d,v) in zip(dd,z)])
  return sum(d2)/sum(d3) + a
end


project(p::HalfSpace,x) = x∈p ? x : x + (p.b - p.a⋅x) * p.a/norm(p.a)

function project(ell::Ellipsoid,y)
  if(y in ell) return y end
  c = center(ell)
  M = shape_matrix(ell)
  x = y-c
  d1,Q = eigen(M)
  d = 1 ./ d1
  z = Q'*x
  a = 0.0
  # Apply Newton's method to solve the problem for the appropriate value:
  while abs(g(d,z,a))>1e-10
    a = dg(d,z,a)
  end
  return Q*Diagonal(1 ./(a*d .+ 1))*Q*x + c
end

## Define approximate projections and reflections.


"""
  Apply the approximate Douglas Rachford algorithm between the sets `A` and `B`.
"""
function aDR(x,A,B,ε)
  if ε ≤ 0  return DR(x,A,B) end
  d = 10.
  yA = an_element(A)
  yB = an_element(B)
  #while y ∉ A∩B
  V = NTuple{4,Vector{Float64}}[]
  nit = 0 
  while d/ε > 1e-6 && nit<200
    # First Project on A
    yA = aproject(A,x; x= yA,pred = (x,g,s,i) -> g⋅(s-x) < d)
    # Then Reflect
    y = 2yA - x 
    # Then Project on B
    yB = aproject(B,y;x = yB,pred = (x,g,s,i) -> g⋅(s-x) < d)
    # Add the iterates to the list.
    push!(V,(y,x,yA,yB))
    # Then Reflect.
    x = 0.5*(x + 2yB - y)
    # Set the new tolerance:
    d = (yA - yB)⋅(yA - yB)*ε
    nit += 1
  end
  return V
end

"""
  Apply the exact Douglas Rachford algorithm between the sets `A` and `B`.
"""
function DR(x,A,B)
  d = 10.
  yA = an_element(A)
  yB = an_element(B)
  #while y ∉ A∩B
  V = NTuple{4,Vector{Float64}}[]
  nit = 0 
  while d > 1e-6 && nit<300
    # First Project on A
    yA = project(A,x)
    # Then Reflect
    y = 2yA - x 
    # Then Project on B
    yB = project(B,y)
    # Add the iterates to the list.
    push!(V,(y,x,yA,yB))
    # Then Reflect.
    x = 0.5*(x + 2yB - y)
    # Set the new tolerance:
    d = (yA - yB)⋅(yA - yB)
    nit += 1
  end
  return V
end
