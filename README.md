# Diffusion model for control and planning tutorial



## Recap of the diffusion model

We learn a score function (similar to the noise direction) and use that to recover our distribution.

Emperical success:
* handling multimodal action distributions
* being suitable for high-dimensional action spaces
* exhibiting impressive training stability

## Movitation: why do we need a diffuser?

Overall objective: use a diffuser as a powerful distribution matching tool for control and planning problems.

Where do we need a distribution match?
1. **Imitation learning**: match the expert's action distribution (mentioning GAIL, adverserial training. with diffusion model, it become more stable. especially for multi-task)
2. **Offline reinforcement learning**: match the policy's action distribution (need to be expressive enough to match the distribution of the policy and also not deviate too much from the expert's distribution, extrapolation error problem)
   * challenge: extrapolation error problem
   * current solution: panelize/constrain OOD samples -> overconserative
3. **Model-based reinforcement learning**: match the dynamic model (need to work in the long horizon) + policy's action distribution(sometimes)

Why diffusion works here?
1. non-autoregressive (no sequential dependency): compounding error is not a problem, but still can generate any length of sequence with certain architecture choise
2. multimodal: can handle multimodal action distributions
3. matching the distribution: can match the distribution of the expert's action
4. High capacity + high expressiveness: can handle high-dimensional action spaces -> foundation models, 50 demostrations per task

smooth

## Practice: how to use diffuser?

Things to diffuse: 
* in image: 2d pixel value
* in control: 1d control/trajectory sequence 

Architecture:
* temporal convolutional network (TCN)

How to make it condition on certain objective?
1. guidance function: directly shift the distribution / cost or learned value etc. 
2. inpainting: fill the missing part of the distribution so as to constrains certain part of the distribution

## Applications: research progress in diffuser for control and planning


## Limitations: what are the challenges?
