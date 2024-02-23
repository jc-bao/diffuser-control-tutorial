# Diffusion model for control and planning tutorial

Tutorial outline:

1. Recap: what is a diffusion model / what problem does it solve?
2. Motivation: why do we need a diffuser in control and planning?
3. Practice: how to use a diffuser in control and planning?
4. Literatures: recent research in a diffuser for control and planning
5. Limitations: what are the remaining challenges?
6. Summary

## Recap: diffusion model

We learn a score function (similar to the noise direction) and use that to recover our distribution.
Empirical success:

![diffusion examples](figs/dalle.png)

* handling multimodal action distributions
* being suitable for high-dimensional action spaces
* exhibiting impressive training stability

## Movitation: why do we need a diffuser?

Overall objective: use a diffuser as a powerful distribution matching tool for control and planning problems.

Where do we need a distribution match?

1. **Imitation learning**: match the expert's action distribution (mentioning GAIL, adverserial training. with diffusion model, it become more stable. especially for multi-task)
   * key problem: matching a high-dimensional action distribution with a small number of optimal demonstrations
   * challenge: powerful/expressive distribution matching tool
2. **Offline reinforcement learning**: match the policy's action distribution (need to be expressive enough to match the distribution of the policy and also not deviate too much from the expert's distribution, extrapolation error problem)
   * key problem: become better than the data with a large number of demonstrations
   * challenge: extrapolation error problem
   * current solution: panelize/constrain OOD samples -> overconserative
3. **Model-based reinforcement learning**: match the dynamic model (need to work in the long horizon) + policy's action distribution(sometimes)

Why does diffusion work here?
7. non-autoregressive (no sequential dependency): compounding error is not a problem, but still can generate any length of sequence with certain architecture choices
8. multimodal: can handle multimodal action distributions
9. matching the distribution: can match the distribution of the expert's action
10. High capacity + high expressiveness: can handle high-dimensional action spaces -> foundation models, 50 demonstrations per task
11. stable training: with a sound mathematical foundation and standard training procedure via multi-stage diffusion, it is stable to train

smooth

## Practice: how to use diffuser?

![diffusion for control overview](figs/planner_policy_model.png)

Things to diffuse: trajectory sequence as single channel image. 

![Diffusion model diffuse future state-action sequence](figs/traj.png)

![Diffuse process for next trajectory](figs/traj_diffuse.png)

Architecture choices:

* temporal convolutional network (TCN)

How to make it condition on certain objectives?

1. guidance function: directly shift the distribution / cost or learned value etc or train a discriminator (classifier) to get the guidance function.

$$
\begin{aligned}
\nabla \log p\left(\boldsymbol{x}_t \mid y\right) & =\nabla \log \left(\frac{p\left(\boldsymbol{x}_t\right) p\left(y \mid \boldsymbol{x}_t\right)}{p(y)}\right) \\
& =\nabla \log p\left(\boldsymbol{x}_t\right)+\nabla \log p\left(y \mid \boldsymbol{x}_t\right)-\nabla \log p(y) \\
& =\underbrace{\nabla \log p\left(\boldsymbol{x}_t\right)}_{\text {unconditional score }}+\underbrace{\nabla \log p\left(y \mid \boldsymbol{x}_t\right)}_{\text {adversarial gradient }}
\end{aligned}
$$

2. classifier-free method: does not need to train a classifier to get the guidance function, now two terms can be interpreted as unconditional score and conditional score.

$$
\begin{aligned}
\nabla \log p\left(\boldsymbol{x}_t \mid y\right) & =\nabla \log p\left(\boldsymbol{x}_t\right)+\gamma\left(\nabla \log p\left(\boldsymbol{x}_t \mid y\right)-\nabla \log p\left(\boldsymbol{x}_t\right)\right) \\
& =\nabla \log p\left(\boldsymbol{x}_t\right)+\gamma \nabla \log p\left(\boldsymbol{x}_t \mid y\right)-\gamma \nabla \log p\left(\boldsymbol{x}_t\right) \\
& =\underbrace{\gamma \nabla \log p\left(\boldsymbol{x}_t \mid y\right)}_{\text {conditional score }}+\underbrace{(1-\gamma) \nabla \log p\left(\boldsymbol{x}_t\right)}_{\text {unconditional score }}
\end{aligned}
$$

3. inpainting: fill the missing part of the distribution so as to constrains certain part of the distribution

## Literatures: research progress in a diffuser for control and planning

Axis1: how to get the score function

* Data-driven: learning from data by manually adding noise to the data
* Hybrid: learning from other optimization process
* Model-based: calculate analytical from the model

Axis2: what to diffuse

* action: directly diffuse for the next action
* state: learn the model
* state-sequence: diffuse for the next state sequence, or sometimes state-action sequence, or sometimes action sequence (for position control)

## Limitations: what are the challenges?

1. computational cost: The diffusion model needs a longer time to train (a few GPUs hours compared with tens of minutes) and inference (iterative sample steps compared with one forward pass). This makes high frequency control and planning difficult to use diffusion model.
2. handle shifting distribution: in online RL, the distribution of the policy will shift keep changing while adapting diffusion model to the new distribution need large amount of data and long time to train. This limit diffusion model to be trained in a fixed rather than dynamic dataset.
3. high variance: depends on initial guess and random sampling, the variance of the diffusion model is high, which limits its application in high precision or safety-critical tasks.
4. constrain satisfaction: the diffusion model is not guaranteed to satisfy the constraints, especially when tested in a constrain different from the training set. This limits its application in adapting to new constraints and tasks.

## Summary

* Why diffusion: stable training, multimodal (learn the field v.s. learn the gradient), scalable

## TODO

1. tutorial on a simple control problem
