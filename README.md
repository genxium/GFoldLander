This is the source code of our paper [A Fuel-Optimal Landing Guidance and Inverse Kinematics Coupled PID Control Solution for Power-Descent Vertical Landing in Simulation](http://arqiipubl.com/ojs/index.php/AMS_Journal/article/view/354), a copy of the paper with typo corrections is also attached in this repo. 

Regardless of many TODO items still pending in the current implementation, we decided to go open source aiming at helping people (especially students) get an easy start on this topic -- you can refer to this repo for setting up the simulation environment and our algorithm for basic closed-loop control implementation :)  

Happy simulating!

### Prerequisites
Please ensure that you have python version >=3.8 but <3.11 installed (either on OSX or Windows, this is to ensure that we have same behaviour for `scipy-1.7.1` solvers).

```
pip3 install -r requirements.txt
```

### Checking Functionality

```
python3 lunarlander_pid_only.py
```

To export math typesetting in draw.io files correctly, please ensure that you have draw.io-16.0.2 or later installed.


### On changing the `env.SCALE`

Kindly note that you might encounter cases where the `pid only` controller doesn't work to land the lander. The feasibility of `pid only` approach is highly impacted by the `Tsettle`, because it's bound directly for the target position. If `Tsettle` were not estimated near a possible value the controller would've output impossible values to achieve and prohibited by plant clipping. Therefore a change on `env.SCALE` or `start position` could make `pid only` infeasible. 

The gfold approach on the other hand only needs an "upper bound for `Tsettle`" and is very flexible for change in `env.SCALE` or `start position`.

It's also noticeable that a change of the `env.SCALE` has much stronger impact on `Iz ~ m*(R^2)` than `m` due to constant density of the lander, thus might require some tuning of `rho_1`, `rho_2`, `rho_side_1` and `rho_side_2` after such change. 
