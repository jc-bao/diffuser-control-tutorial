# Diffusion model for control and planning tutorial

Tutorial outline:

1. Recap: what is a diffusion model / what problem does it solve?
2. Motivation: why do we need a diffuser in control and planning?
3. Practice: how to use a diffuser in control and planning?
4. Literatures: recent research in a diffuser for control and planning
5. Summary & Limitations: what we can do and what we cannot do

## Recap: diffusion model

Diffusion model is a generative model that can generate samples from a given distribution. It is a powerful distribution matching tool that can match the distribution of the dataset, which is widely used in image generation, text generation, and other generative tasks.

![diffusion examples](figs/dalle.png)

The key component of the diffusion model is the score function, which is the noise direction that can be used to update the sample to match the distribution. By learning the score function, the diffusion model can generate samples from the given distribution.

$$
\boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_i+c \nabla \log p\left(\boldsymbol{x}_i\right)+\sqrt{2 c} \boldsymbol{\epsilon}, \quad i=0,1, \ldots, K
$$

![Annealed Langevin dynamics combine a sequence of Langevin chains with gradually decreasing noise scales.](figs/distribution_match.gif)

The unique properties of the diffusion model include:

* **Multimodal**: It can handle multimodal distributions, which is hard to learn my directly predicting the distribution.
* **Scalable**: The method can be scaled to high-dimensional distribution matching problems. 
* **Stable**: With a sound mathematical foundation and standard training procedure via multi-stage diffusion, it is stable to train.
* **Non-autoregressive**: It can handle non-autoregressive and multimodal distribution matching by predicting the whole trajectory sequence at once.


## Motivation: why do we need a diffuser in control and planning?

> compare with other generative models
> what to learn here

From the control and planning perspective, there are lots of scenarios where we need to match the distribution of the dataset, such as: 

| Scenario | Challenge | Solution |
| --- | --- | --- |
| Imitation learning | Match the expert's action distribution with limited data. Common method like GAIL using adversarial training to match the distribution. BC cannot handle multimodal distribution. | Diffusion model can matching the distribution of the expert's action with high capacity and high expressiveness. |
| Offline reinforcement learning | Perform better than dataset with a large number of demonstrations. Here need to make sure the policy's action distribution is close to the dataset while improving the performance. Common method like CQL penalize OOD samples, make the method overconserative. | Diffusion model can match the dataset's action and regularize the policy's action distribution. |
| Model-based reinforcement learning | Match the dynamic model and (sometimes) policy's action distribution. First learning the model and then use the model to plan in a auto-regressive manner. This method suffers from compounding error. | Diffusion model can handle non-autoregressive and multimodal distribution matching by predicting the whole trajectory sequence at once. |

## Practice: how to use the diffuser?

### What to diffuse?
By concatenating the state and action, we can diffuse the state-action sequence, which is like diffusing a single-channel image.

|Task|Thing's to diffuse|How to diffuse|
|---|---|---|
|Image generation|![noise image](figs/denoise_image.gif)|![image diffuse](figs/image_diffuse.jpg)|
|Planning|![Diffusion model diffuse future state-action sequence](figs/denoise_traj.gif)|![Diffuse process for next trajectory](figs/traj_diffuse.png)|


### How to impose constraints/objectives?

**guidance function**

directly shift the distribution/cost or learned value etc or train a discriminator (classifier) to get the guidance function.

Predefined the guidance function:

$$
\tilde{p}_\theta(\boldsymbol{\tau}) \propto p_\theta(\boldsymbol{\tau}) h(\boldsymbol{\tau})
$$

Learned guidance function:

$$
\begin{aligned}
\nabla \log p\left(\boldsymbol{x}_t \mid y\right) & =\nabla \log \left(\frac{p\left(\boldsymbol{x}_t\right) p\left(y \mid \boldsymbol{x}_t\right)}{p(y)}\right) \\
& =\nabla \log p\left(\boldsymbol{x}_t\right)+\nabla \log p\left(y \mid \boldsymbol{x}_t\right)-\nabla \log p(y) \\
& =\underbrace{\nabla \log p\left(\boldsymbol{x}_t\right)}_{\text {unconditional score }}+\underbrace{\nabla \log p\left(y \mid \boldsymbol{x}_t\right)}_{\text {adversarial gradient }}
\end{aligned}
$$

Possible issue: the distribution is pushing out the distribution of the data, which is not what we want.

**classifier-free method**

Does not need to train a classifier to get the guidance function, now two terms can be interpreted as unconditional score and conditional score.

$$
\begin{aligned}
\nabla \log p\left(\boldsymbol{x}_t \mid y\right) & =\nabla \log p\left(\boldsymbol{x}_t\right)+\gamma\left(\nabla \log p\left(\boldsymbol{x}_t \mid y\right)-\nabla \log p\left(\boldsymbol{x}_t\right)\right) \\
& =\nabla \log p\left(\boldsymbol{x}_t\right)+\gamma \nabla \log p\left(\boldsymbol{x}_t \mid y\right)-\gamma \nabla \log p\left(\boldsymbol{x}_t\right) \\
& =\underbrace{\gamma \nabla \log p\left(\boldsymbol{x}_t \mid y\right)}_{\text {conditional score }}+\underbrace{(1-\gamma) \nabla \log p\left(\boldsymbol{x}_t\right)}_{\text {unconditional score }}
\end{aligned}
$$

| Guidance function method | Classifier-free method |
| --- | --- |
| ![guidance function](figs/guidance_algo.png) | ![classifier-free guidance](figs/classifierfree_algo.png) |

**inpainting**

If the control problem has specific state constrains, we can just fix the state and fill the missing part of the distribution. This is very useful in goal reaching tasks.

![inpainting](figs/inpaint.gif)

## Literatures: research progress in a diffuser for control and planning

> safe diffuser, aloha, umi paper

$$
\color{red}\nabla_\tau \log 
\color{magenta}P
\color{black}( 
\color{blue}{\tau} | 
\color{green} y
\color{black})
$$

Axis1: how to get the score function

* Data-driven: learning from data by manually adding noise to the data
* Hybrid: learning from other optimization process
* Model-based: calculate analytical from the model

Axis2: what to diffuse

![diffusion for control overview](figs/planner_policy_model.png)

* action: directly diffuse for the next action
* state: learn the model
* state-sequence: diffuse for the next state sequence, or sometimes state-action sequence, or sometimes action sequence (for position control)

## Conclusion & Limitations: what are the challenges?

**How diffusion work**
Compared with learning the explicit policy directly or learning the energy-based model, the diffusion model can handle multimodal distribution and higher-dimensional distribution matching. 

![Compare distribution with other models](figs/policy_vs_diffusion.png)

**Limitations**
1. computational cost: The diffusion model needs a longer time to train (a few GPUs hours compared with tens of minutes) and inference (iterative sample steps compared with one forward pass). This makes high frequency control and planning difficult to use diffusion model.
2. handle shifting distribution: in online RL, the distribution of the policy will shift keep changing while adapting diffusion model to the new distribution need large amount of data and long time to train. This limit diffusion model to be trained in a fixed rather than dynamic dataset.
3. high variance: depends on initial guess and random sampling, the variance of the diffusion model is high, which limits its application in high precision or safety-critical tasks.
4. constrain satisfaction: the diffusion model is not guaranteed to satisfy the constraints, especially when tested in a constrain different from the training set. This limits its application in adapting to new constraints and tasks.

## Summary

* Why diffusion: stable training, multimodal (learn the field v.s. learn the gradient), scalable

## TODO

1. tutorial on a simple control problem
