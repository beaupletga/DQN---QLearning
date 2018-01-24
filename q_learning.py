import numpy as np
np.random.seed(1)


def start_map(num_states,num_actions):
    # R=np.random.rand(num_states,num_actions)
    # R=np.zeros((num_states,num_actions))
    # R[2,-1]=100

    R=np.array([
    [-1,-1,-1,-1,0,-1],
    [-1,-1,-1,0,-1,100],
    [-1,-1,-1,0,-1,-1],
    [-1,0,0,-1,0,-1],
    [0,-1,-1,0,-1,100],
    [-1,0,-1,-1,0,100]
    ])

    Q=np.zeros((num_states,num_actions))

    return R,Q


def play_game(R,Q):
    state=np.random.randint(0,R.shape[0]-1)
    reward=0
    states_list=[state]

    while state!=R.shape[0]-1:
        next_state=np.argmax(Q[state])
        states_list.append(next_state)
        reward+=R[state,next_state]
        state=next_state

    print "Reward :",reward
    print "States List :",states_list


def main():

    num_states=6
    num_actions=6
    R,Q=start_map(num_states,num_actions)

    lr=0.99
    gamma=0.5

    # to avoid beeing greedy and enable to take lower rewarded action
    eps=0.05

    iter=0
    while iter<1000:
        state=0
        action=0
        while True:
            # greedy (we take the most rewarded action)
            # a=[[state,j] for j in range(num_actions) if Q[state,j]==np.amax(Q[state])]
            # next_state=a[np.random.randint(0,len(a))][1]

            if np.random.rand()<=eps:
                next_state=np.random.randint(0,num_actions)
            else:
                next_state=np.argmax(Q[state])

            Q[state,next_state]=Q[state,next_state]+lr*(R[state,next_state]+gamma*np.amax(Q[next_state])-Q[state,next_state])

            if state==num_states-1:
                break
            state=next_state
        iter+=1
    print Q
    play_game(R,Q)

main()
