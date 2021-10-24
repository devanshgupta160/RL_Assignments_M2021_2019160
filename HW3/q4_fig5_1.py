from random import randint
import numpy as np
import matplotlib.pyplot as plt

def draw():
    return min(randint(1, 13), 10)

def getPlayerAction(state):
    if(state[0] == 20 or state[0] == 21):
        return "stick"
    return "hit"

def getVal(cards):
    val = 0	
    aux = 0
    useAce = False
    
    for c in cards:
        val += c
        aux += c
        if(c == 1) and not useAce:
            useAce = True
            aux += 10
    
    if(aux <= 21):
        return aux, useAce
    
    return val, False


def blackjack(num_games, state_val, state_count):
    for _ in range(num_games):
        states = []
        rewards = [None]

        player = [draw(), draw()]
        d_faceup = draw()
        dealer = [d_faceup, draw()]
        pl_val, use_ace = getVal(player)
        curr_state = (pl_val, d_faceup, use_ace)
        terminal = False
        while not terminal:
            action = getPlayerAction(curr_state)
            terminal = False
            g_end = False
            if action == "hit":
                player.append(draw())
            elif action == "stick":
                v_d, _ = getVal(dealer)
                while v_d <= 17:
                    dealer.append(draw())
                    v_d, _ = getVal(dealer)
                g_end = True
            
            pl_val, ace_ = getVal(player)
            deal_val, _ = getVal(dealer)

            reward = 0
            terminal = (pl_val > 21) or g_end
            if not g_end:
                reward = -1 if pl_val > 21 else 0
            else:
                if deal_val > 21 or deal_val < pl_val:
                    reward = 1
                elif deal_val == pl_val:
                    reward = 0
                else:
                    reward = -1
            
            next_state = (pl_val, d_faceup, ace_)
            states.append((pl_val, d_faceup, ace_))
            rewards.append(reward)
            curr_state = next_state
        states.append(next_state)
        G = 0
        t_end = len(rewards) - 2
        for i in range(t_end, -1, -1):
            G += rewards[i+1]
            St = states[i]
            if St not in state_val.keys():
                state_val[St] = 0
                state_count[St] = 0
            state_val[St] += G
            state_count[St] += 1
        
    for s in state_val.keys():
        state_val[s] /= state_count[s]

    return state_val, state_count

def plotSV(state_val, n):
    grid_wace = np.zeros((10, 10))
    grid_woace = np.zeros((10, 10))

    for s in state_val.keys():
        pl_val, d_faceup, ace_ = s
        if pl_val > 21 or pl_val < 11:
                continue
        if ace_:
            grid_wace[pl_val - 12][d_faceup - 1] = state_val[s]
        else:
            grid_woace[pl_val - 12][d_faceup - 1] = state_val[s]
        
    y = [12 + i for i in range(10)]
    x = [i for i in range(1, 11)]

    xx, yy = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.plot_surface(xx, yy, grid_wace)
    plt.title(f"With Usable Ace - {n} Episodes")
    plt.xlabel("Dealer Showing Card Value")
    plt.ylabel("Player Value")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.plot_surface(xx, yy, grid_woace)
    plt.title(f"Without Usable Ace - {n} Episodes")
    plt.xlabel("Dealer Showing Card Value")
    plt.ylabel("Player Value")
    plt.show()

state_val = {}
state_count = {}

state_val, state_count = blackjack(10000, state_val, state_count)
plotSV(state_val, 10000)
state_val, state_count = blackjack(490000, state_val, state_count)
plotSV(state_val, 500000)
