# SARSA Learning Algorithm


## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## SARSA LEARNING ALGORITHM
Include the steps involved in the SARSA Learning algorithm

## SARSA LEARNING FUNCTION
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    # Write your code here
    select_action = lambda state, Q, epsilon:np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas =decay_schedule(init_alpha,min_alpha, alpha_decay_ratio,n_episodes)
    epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done = env.reset(), False
      action =select_action(state,Q,epsilons[e])
      while not done:
        next_state, reward, done, _=env.step(action)
        next_action=select_action(next_state,Q,epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target- Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state,action =next_state,next_action
      Q_track[e]=Q
      pi_track.append(np.argmax(Q, axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

## OUTPUT:
Optimal policy:
![image](https://github.com/gpavithra673/sarsa-learning/assets/93427264/6f3f03c6-235a-4610-8be3-427b267f0fea)
FVMC action-value function:
![image](https://github.com/gpavithra673/sarsa-learning/assets/93427264/919688e8-38bb-490d-8f15-1014f5cbcefa)
SARSA action-values function:
![image](https://github.com/gpavithra673/sarsa-learning/assets/93427264/1c7d1f6d-6fc4-44a8-89dd-cea3097b036d)
Comparison between FVMS vs SARSA:
![image](https://github.com/gpavithra673/sarsa-learning/assets/93427264/4a91c14d-3e2c-421f-a3a1-5ed2b9d8523e)
![image](https://github.com/gpavithra673/sarsa-learning/assets/93427264/ad766452-8240-4514-b91c-e748ffe75413)

## RESULT:

Write your result here
