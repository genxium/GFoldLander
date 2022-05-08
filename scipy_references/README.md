# 1. Minimization with constraints
In general, a minimization objective with inequality constraints can be phrased in the following form
```
minimize    f(x)
subject to  h(x) = 0
            g(x) <= 0
```

where 
- `x` here is an n-dimensional vector input;
- `f(x)`, `h(x)` and `g(x)` are "not necessarily linear";
- `h(x)` and `g(x)` can be vector valued.

## 1.1 Minimization with only equality constraints
If the problem is with only equality constraints
```
minimize    f(x)
subject to  h(x) = 0
```
it can be solved simply using `basic Lagrangian Multiplier method`, which is clearly visualized by [this Khan Academy Tutorial](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/constrained-optimization/a/lagrange-multipliers-single-constraint).

## 1.2 Minimization with mixed constraints
If the problem is with mixed equality and inequality constraints, there're at least 2 methods to figure out an answer.

### 1.2.1 Penalty method
The constrained problem can be converted into an unconstrained problem by adding each constraint to the objective as a "Penalty Function". See
- [this lecture from Christopher Lum](https://pan.baidu.com/s/1yNlnLtd-VBKgDBxUhqFudg) (zuoa), and
- search for "log barrier" in [this book from Boyd & Vandenberghe](https://pan.baidu.com/s/1GoY2yGgDZfAjaJiQb4_Mhw) (rnqq)
for more information. Kindly note that though the book is dedicated for convex programming, the part for "Penalty Function" is generic.

Pros
- no requirement for the problem to be "regular", i.e. no "constraint qualification"

Cons
- no guarantee on finding the solution which satisfies all constraints if one existed, e.g. depending on initial guess chosen

### 1.2.2 KKT conditions method
Watch 
- [this video lecture for theoretical development](https://pan.baidu.com/s/1f6LrcWjPorqUnSCwYqA4gA) (3oof), and then
- [this video lecture for an example](https://pan.baidu.com/s/1ocoJ-pxBiSQPJHpq0F4rKg) (ac8d).

Although during the theoretical development some coefficients are introduced to take constraints into objective, making it a little similar to `1.2.1`, _the aftermath is quite different_. 

Kindly note that `KKT conditions method` is only applicable to `regular problems` which satisfy `some constraint qualifications`. The term `some constraint qualifications` might seem vague at a first glance but it should be analyzed case by case.     

Pros
- guarantee on finding the solution which satisfies all constraints if one existed

Cons
- there're requirements of `constraint qualifications` for the problem to be "regular", i.e. such that the `duality gap = 0`

### 1.2.3 Equality conversion method
Yet another method being a variant of `1.2.2` is by introducing `Slack Variables` to convert inequality constraints into equalities, i.e. 
```
g[i](x) + s[i]^2 = 0
```
, then apply `1.1` to seek solution for all possible combinations of `subset(g(x)) = 0`. 

Pros
- guarantee on finding the solution which satisfies all constraints if one existed

Cons
- there're requirements of `constraint qualifications` for the problem to be "regular", i.e. such that the `duality gap = 0`
- need traverse all possible combinations of `subset(g(x)) = 0`

# 2. SLSQP

With all of `section 1` understood, we might have an objective `f(x)` too complicated for finding the gradient analytically, therefore it's possible to approximate `f(x)` locally as a `quadratic objective`, then apply the methods from `section 1` sequentially (i.e. many times) to find an answer. Please watch [this tutorial](https://pan.baidu.com/s/1Md9sl5gzTbQRz983sY_dNg) (nwxj) for how "unconstrained SLSQP" works in example. 

I somehow found this [C translation for the SLSQP impl](http://mad.web.cern.ch/mad/releases/madng/madng-git/lib/nlopt/src/algs/slsqp/slsqp.c), where the [original impl is by f-lang](https://github.com/scipy/scipy/blob/main/scipy/optimize/slsqp/slsqp_optmz.f).


