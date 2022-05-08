# 1. CartPole balancing
Introducing the ["CartPole balancing" game](https://jeffjar.me/cartpole.html). The purpose of this game is merely to keep the pole balanced/erected, REGARDLESS OF the position the cart is at (unless you're out of bounds at the left or right extreme).  

For this simple purpose, you can do a simulation by using OpenAI's [CartPole gym](https://gym.openai.com/envs/CartPole-v1/), and a simplest PID controller as shown in https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c.

# 2. CartPole moving to point target
A meaningful CartPole game is to move the CartPole to a specified position on the horizontal axis, as demonstrated in [Steve Brunton's lecture](https://www.youtube.com/watch?v=1_UobILf3cc), where the solution to this improved problem is shown around 00:11:40. See https://pan.baidu.com/s/1JZLJCgxqFizXY_quJY8vMQ (rgl4) instead if you have difficulty accessing YouTube.

Steve used "LQR controller" to fulfill the needs, yet this algorithm is limited to "linear quadratic" model of the plant, where many plants of interest are not of this type. If you're confused with the jargons, 
- `plant` means the real, physical dynamics of something, e.g. the CartPole or the LunarLander, often with INEVITABLE interactions with the wind or water, and
- `model` means the mathematical model we use to estimate the `plant`, often ignoring many subtle interactions with wind or simplifying high-order interactions with water at low velocity. 

"PID controller" on the other hand is flexible about the model, yet only easy when the `model` is "Single Input Single Output"(a.k.a. `SISO`), which is true for the CartPole balancing problem but not the CartPole moving problem. See [here](https://github.com/genxium/lunarlander/blob/master/mini_tutorials/pid_control_getting_started.md) for a more detailed description. 

Build a "PID controller" to move the CartPole to a point target (on the horizontal axis) while keeping the pole erected/balanced within a 5 degrees swing range first.

# 3. LunarLander moving to point target 
Done in [this script](./lunarlander_pid_only.py).

# 4. LunarLander G-FOLD path planning and following
Done in [this script](./lunarlander_offline_gfold.py).

# 5. Optimize the cvx programming algorithm
- [x] In [this script](./lunarlander_offline_gfold.py), use [SOC constraints](https://www.cvxpy.org/examples/basic/socp.html) whenever possible to hint and speed up the solution! 
    - Note that SOCP is NOT [QP](https://www.cvxpy.org/examples/basic/quadratic_program.html), where the latter only considers linear constraints, i.e. QP is a subset of SOCP.
    - Checkout the [Disciplined Convex Programming](https://dcp.stanford.edu/) rules and quiz to see how cvxpy knows the curvature and sign characteristics of an expression/constraint.
- [] In [the plant](./environments/lunarlanderfuel.py), try to simulate "air drag", refer to [this article](https://imranedu.wordpress.com/2015/01/07/how-to-simulate-realistic-air-friction-in-box2d-starling-version/) and [this project(Ex4.4)](https://github.com/genxium/dynamic-optimization).
- [] In [this script](./lunarlander_offline_gfold.py), replace the `z = ln(mass)` symbol with `lm = ln(mass)` and note this change in the comment. 
