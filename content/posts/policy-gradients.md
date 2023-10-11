---
author: AUTHOR NAME
date: '2021-07-20T13:56:01-08:00'
type: post
title: 'Policy Gradients'
subtitle: ''
toc: true
math: true
categories:
  - math
  - rl
---
Policy-based RL is a method in which a parameterized policy is directly optimized to select an action. This is different from the
methods that estimate the action-value function `$Q(s, a)$`, from which a policy is derived either by doing a argmax or 
softmax over the action-value function. The advantages of optimizing the policy is numerous especially in settings where
approximating the action-value function is difficult like in continuous state and action spaces. Adopting stochastic policies often
needed for games POMDP environments, where the optimal play might be two different things with specific probabilities, is much simpler in policy
based RL[^2].

<!--more-->

## The Policy Gradient Theorem

The goal of policy-based RL is to find parameters `$\theta$` that maximises the expected reward. 

`$$
\theta^* = \argmax\limits_{\theta}\mathbb{E}_{\tau \sim p_{\theta(\tau)}}\lbrack\sum_t r(s_t, a_t) \rbrack
$$`

The objective function is 

`$$
\mathnormal{J(\theta)} = \mathbb{E}_{\tau \sim p_{\theta(\tau)}}\lbrack\sum_t r(s_t, a_t) \rbrack
$$`

In terms of a PDF this can be written as

`$$
\mathnormal{J(\theta)} = \int \mathnormal{p_\theta(\tau)r(\tau)d\tau}
$$`

Taking derivative wrt to `$\theta$` on both sides

`$$
\tag{1} \triangledown_\theta\mathnormal{J(\theta)} = \int \triangledown_\theta\mathnormal{p_\theta(\tau)r(\tau)d\tau}
$$`

Consider the term
`$$
\triangledown_\theta\mathnormal{p_\theta(\tau)}
$$`

Multiply and divide by `$\mathnormal{p_\theta(\tau)}$`

`$$
\tag{2} \mathnormal{p_\theta(\tau) \frac{\triangledown_\theta p_\theta(\tau)}{p_\theta(\tau)}} = \mathnormal{p_\theta(\tau)\triangledown_\theta \log p_\theta(\tau)}
$$`

Subsituting `$(2)\space in \space (1)$`

`$$
\triangledown_\theta\mathnormal{J(\theta)} = \int \mathnormal{p_\theta(\tau)\triangledown_\theta \log p_\theta(\tau)r(\tau)d\tau }
$$`

Converting back to expectation

`$$
\tag{3} \triangledown_\theta\mathnormal{J(\theta)} =  \mathbb{E}_{\tau \sim p_{\theta(\tau)}}\lbrack \triangledown_\theta \log p_\theta(\tau)r(\tau)\rbrack
$$`

Unrolling `$p_\theta(\tau)$` we get (see Appendix A for more details)

`$$
p_\theta(\tau) = p_\theta(s_1, a_1, ..., s_T, a_T) = p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t | s_t) p(s_{t+1}|s_t, a_t)
$$`
Taking log on both sides

`$$
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^{T} \log \pi_\theta(a_t | s_t)  + \log p(s_{t+1}|s_t, a_t)
$$`

Taking the derivative wrt to `$\theta$`

`$$
\triangledown_\theta\log p_\theta(\tau) = \triangledown_\theta\log p(s_1) + \sum_{t=1}^{T} \triangledown_\theta \log \pi_\theta(a_t | s_t)  + \triangledown_\theta \log p(s_{t+1}|s_t, a_t)
$$`

Since both `$\log p(s_1)$` and `$\log p(s_{t+1}|s_t, a_t)$` are not dependent on `$\theta$` their derivative wrt to theta will be zero.

`$$
\tag{4} \triangledown_\theta\log p_\theta(\tau) = \sum_{t=1}^{T} \triangledown_\theta \log \pi_\theta(a_t | s_t) 
$$`

Subsituting `$(4) \space in \space (3)$`
`$$
\triangledown_\theta\mathnormal{J(\theta)} =  \mathbb{E}_{\tau \sim p_{\theta(\tau)}}\biggl\lbrack \biggl\lparen \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t) \biggr\rparen  \biggl\lparen \sum_{t=1}^{T} r(s_t, a_t)\biggr\rparen   \biggr\rbrack
$$`

This shows that the gradient with respect to the policy parameters does not involve the derivative of the state distribution.

## Baselines 

Policy gradients suffer from high variance unless using a large number of samples. One way to reduce the variance is to
subtract the sum of the rewards by the average reward. Intuitively, this would give an higher weight to trajectories that 
lead to better than average reward and  supress trajectories that lead to worse than average reward[^1]. 

The reason this is allowed is because it still leads to an unbiased estimate of the cost function as shown below

`$$
\triangledown_\theta\mathnormal{J(\theta)} =  \mathbb{E}_{\tau \sim p_{\theta(\tau)}}\biggl\lbrack \biggl\lparen \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t) \biggr\rparen  \biggl\lparen \sum_{t=1}^{T} r(s_t, a_t)\biggr\rparen   \biggr\rbrack
$$`

Since we use the Monte Carlo estimates 

`$$
\triangledown_\theta\mathnormal{J(\theta)} \approx \frac{1}{N} \sum_{i=1}^{N} \biggl\lbrack \biggl\lparen \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t) \biggr\rparen  \biggl\lparen \sum_{t=1}^{T} r(s_t, a_t)\biggr\rparen \biggr\rbrack
$$`

`$$
\triangledown_\theta\mathnormal{J(\theta)} \approx \frac{1}{N} \sum_{i=1}^{N} \biggl\lbrack \biggl\lparen \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t) \biggr\rparen  \biggl\lparen \sum_{t=1}^{T} r(s_t, a_t) - b\biggr\rparen \biggr\rbrack
$$`

where b is 

`$$
 b = \frac{1}{N} \sum_{i=1}^{N} \biggl\lparen \sum_{t=1}^{T} r(s_t, a_t)\biggr\rparen   
$$`

To make the notations more legible we will refer the reward the policy gradient over trajectories as a function of `$ \tau$`. So, 

`$$
 r(\tau) = \sum_{t=1}^{T} r(s_t, a_t)
$$`
and 

`$$
\triangledown_\theta \log p_\theta(\tau) = \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t)
$$`

`$$
\triangledown_\theta\mathnormal{J(\theta)} \approx \frac{1}{N} \sum_{i=1}^{N} \biggl\lbrack \biggl\lparen \triangledown_\theta \log p_\theta(\tau)  \biggr\rparen  \biggl\lparen r(\tau) - b\biggr\rparen   \biggr\rbrack
$$`

`$$
\triangledown_\theta\mathnormal{J(\theta)} \approx \frac{1}{N} \sum_{i=1}^{N} \biggl\lbrack  \triangledown_\theta \log p_\theta(\tau) r(\tau) - \triangledown_\theta \log p_\theta(\tau) b\biggr\rparen   \biggr\rbrack
$$`

First term is the original policy gradient term, and let's take a closer look at the second term

`$$
\tag{5} \mathbb{E}(\triangledown_\theta \log p_\theta(\tau))b = \int p_\theta(\tau) \triangledown_\theta \log p_\theta(\tau)b d\tau
$$`


As shown earlier `$ p_\theta(\tau)\triangledown_\theta \log p_\theta(\tau)$` can be written as `$ \triangledown_\theta p_\theta(\tau) $`[^1]. Subsituting this in (5), 

{{< marginnote >}} Whether this interchange of integral and derivative is allowed in all cases needs further consideration. Please refer https://pages.stat.wisc.edu/~shao/stat609/stat609-07.pdf   {{< /marginnote >}}
`$$
\int \triangledown_\theta p_\theta(\tau)b d\tau = \triangledown_\theta b \int p_\theta(\tau) d\tau = \triangledown_\theta b = 0
$$`

This shows that the subtraction of the baseline does not affect our estimates of the cost function although it does reduce it's variance. In practice, since we take the 
Monte Carlo estimates and this might introduce some bias.

## REINFORCE 

Calculate the Monte Carlo estimate of the gradient of the cost function by sampling trajectories and then update the parameters of the policy
using a learning rate `$\alpha$`. This can be interpreted as: it reinforces the good trajectories since there would be a larger gradient updates
for these trajectories[^3].


| ALGORITHM                                                                                                              |
|------------------------------------------------------------------------------------------------------------------------|
| 1. Initialize policy parameters `$\theta$`                                                                             |
| 2. Run policy `$\pi_\theta$` and collect trajectories `$\tau = (\tau_0, \tau_1 \cdots \tau_n) $`                       |
| 3. Compute baseline using `$ b = \frac{1}{N} \sum_{i=1}^{N} \biggl\lparen \sum_{t=1}^{T} r(s_t, a_t)\biggr\rparen   $` |
| 4. Compute `$ \triangledown_\theta\mathnormal{J(\theta)}$` with b                                                      |
| 5. Update `$ \theta \leftarrow \theta + \alpha\triangledown_\theta\mathnormal{J(\theta)}$`                             |
| 5. Repeat 2. until convergence                                                                                         |

[^1]: [Deep RL Course, UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/)

[^2]: [Sutton & Barto, Reinforcement Learning: An Introduction, 2nd Edition](http://incompleteideas.net/book/the-book-2nd.html)

[^3]: [Kevin P Murphy, Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html)

[^4]: [Schulman, et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

[^5]: [Bellman Operators Stackexchange](https://ai.stackexchange.com/questions/11057/what-is-the-bellman-operator-in-reinforcement-learning)

[^6]: [Rafael Stekolshchik, Entropy in Soft Actor-Critic](https://towardsdatascience.com/entropy-in-soft-actor-critic-part-1-92c2cd3a3515)

[^7] [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)

## Advantage Actor Critic
Instead of using the Monte Carlo rollouts we can use the 1 step return estimate of the value function `$ V(s_t) = r_t + \gamma V(s_{t+1}) $`. Using
our current estimate of the value function as the baseline we can use this in conjuction with the policy grads to get the final update 


`$$
\triangledown_\theta\mathnormal{J(\theta)} \approx   \biggl\lbrack \biggl\lparen \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t) \biggr\rparen  \biggl\lparen \sum_{t=1}^{T} r_t + \gamma V_w(s_{t+1}) - V_w(s_t)\biggr\rparen \biggr\rbrack
$$`

`$$
\triangledown_\theta\mathnormal{J(\theta)} \approx   \biggl\lbrack \biggl\lparen \sum_{t=1}^{T} \triangledown_\theta\log \pi_\theta(a_t | s_t) \biggr\rparen  \biggl\lparen \sum_{t=1}^{T} \delta_t)\biggr\rparen \biggr\rbrack
$$`

where,

`$$
\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)
$$`



| ON-POLICY ALGORITHM                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------|
| 1. Initialize policy parameters `$\theta$` and critic parameters `$w$`                                                     |
| 2. **for** t = 0, 1, 2 ... t; **do**                                                                                             |
| 3. `$\space\space$` Sample  `$ a \sim \pi_\theta(a\vert s) $`                                                                    |
| 3. `$\space\space$` Observe `$ (s, s', a, r) $`                                                                                  |
| 4. `$\space\space$` Compute `$\delta = r + \gamma V_w(s') - V_w(s) $`                                                        |
| 5. `$\space\space$` Update `$ w \leftarrow w + \eta_w \delta \triangledown_w V_w(s) $`                                           |
| 6.  `$\space\space$` Update `$ \theta \leftarrow \theta + \eta_\theta \delta \triangledown_\theta \log \pi_\theta(a \vert s)  $` |
| 7. **end**                                                                                                                               |
| 8. Repeat 2. until convergence                                                                                                   |
|                                                                                                                                  |

The `$ \delta_t $` showe above is for a single setp return, but this could be extended to n-step returns of more than 1 future states
are considered. More genrally, it is defined as

`$$
G_{t:t+n} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1}r_{t+n-1} + r^n V_w(s_{t+n})
$$`

and the n-step advantage which can be replace `$ \delta $` in the above algorithm is as follows:

`$$
A_{\pi_\theta}^{(n)}(s_t, a_t) = G_{t:t+n} - V_w(s)
$$`

The n-step reward is an unbiased estimator but has high variance, and just using the n-step value function gives a lower variance but is biased[^3]. The bias-variance tradeoff can be controlled by changing the value of n, where lower values of n gives a biased low variance estimator and higher values gives a unbiased high variance estimator. Instead, as shown by Schulman et al.[^4] we can take the weighed average of the returns. The choice of using exponential moving average of parameter `$\lambda$` turns out to be a good choice and the average can be shown equivalent to

`$$
A_{\pi_\theta}^{\lambda} = \sum_{l =0}^{\infty} (r \lambda)^l \delta_{t+l}
$$`

where `$ \delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t) $` is the TD error at time t.

## Appendix

### Appendix A

To unroll `$ p_\theta(\tau) = p_\theta(s_1, a_1, ..., s_T, a_T) $` we use the chain rule of probability


`$$
p_\theta(\tau)  =  p_\theta(s_1, a_1, ..., s_T, a_T)
$$`
`$$
p_\theta(\tau)  =  p(s_1)p(a_1 | s_1)p(s_2 | s_1, a_2)p(a_2|s_2, s_1, a_1)p(s_3 | a_3, s_2, s_1, a_1)  \cdots p(s_n|s_{n-1},a_{n-1}, s_{n-2})
$$`

Since we assume that the decision process  is markovian the state `$s_3$` does not depend on the state `$s_1$` or the action
`$a_1$` hence we can rewrite the above as 

`$$
p_\theta(\tau)  =  p(s_1)p(a_1 | s_1)p(s_2 | s_1, a_2)p(a_2|s_2)p(s_3 | a_3, s_2)  \cdots p(s_n | s_{n-1}, a_{n-1})
$$`



`$$
p_\theta(\tau)  = p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t | s_t) p(s_{t+1}|s_t, a_t)
$$`
