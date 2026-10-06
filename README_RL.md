# Simple trainable MegaGem bots

Python 3.10+; no additional packages. Existing rule bots and server files are unchanged.

## Train locally

Run from this directory:

```sh
python3 train_rl.py --method reinforce --games 2000 --model models/reinforce.json
python3 train_rl.py --method mc --games 2000 --model models/mc.json
python3 train_rl.py --resume --games 5000 --model models/reinforce.json
python3 train_rl.py --evaluate --eval-games 1000 --eval-seed 2000000 --model models/reinforce.json
```

Local training imports the existing unpacked server's game engine directly. It uses its real decks, scoring, missions, and starter opponents, with randomly selected opponents and rotating seats. No server process or token is required. `--opponents NAME NAME` restricts the opponent pool; `--help` lists valid names. `--seed` controls training; `--eval-seed` controls separate evaluation deals. Resume restores parameters and the reward baseline, but does not reproduce an uninterrupted random-number stream.

Each 100-game block saves an atomic JSON checkpoint. Evaluation disables learning and exploration, compares against an untrained greedy policy on the same seeds, and writes `MODEL.evaluation.json`. That baseline always bids zero and uses the first legal color: it is intentionally weak, not a competitive rule bot. Opponents can respond differently to different actions despite matched seeds. Use fresh evaluation seeds and several training seeds before drawing strong performance conclusions. Do not run two writers on one checkpoint.

## Play or train through your existing server

Start your server as usual. Supply your existing practice token (or set `ALGOGEM_TOKEN`):

```sh
python3 algogem_rl.py --model models/reinforce.json --server localhost --port 5000 --token YOUR_TOKEN --mode practice
python3 algogem_rl.py --model models/reinforce.json --server localhost --port 5000 --token YOUR_TOKEN --mode practice --train
```

The first command plays a frozen policy. The second explores, learns from each completed game, and saves after each game. Use `--mode pvp` for frozen competition play. Online training is restricted to practice. Reconnect if the server ends the practice session; the saved policy carries over. An interrupted game is discarded. The client uses the original `algogem_client.py` protocol, not the separate `_2` client. Live server connectivity requires your running server and token.

## What is actually learned?

`rl_policy.py` is a small reusable episodic learner: give it a state vector, a decision-head name, legal action IDs, and a final reward. Each action has a learned linear weight vector. `algogem_rl.py` supplies MegaGem observations and maps decisions to bids/reveals.

- **REINFORCE:** a linear softmax policy samples actions during training. After a complete game it applies the log-probability policy gradient, using terminal reward minus a moving-average baseline. This directly learns an action policy.
- **MC:** epsilon-greedy Monte Carlo action-value learning with a linear approximator. After a game, normalized gradient updates move visited action values toward the observed terminal return. It uses neither bootstrapping nor a target network.

Both use undiscounted reward `(own score - mean opponent score) / 100`. Every decision in a game gets that return. This optimizes score margin, not directly win rate. There is no hard-coded bidding strategy, neural network, DQN, replay buffer, or planning simulation. Local game simulation produces real training episodes.

Observations include coins, auction progress/type/value, seat, public balances, own hand and collection, offered gems, revealed colors, the full value chart, remaining mission requirements, debt, investment returns, and earned mission rewards. No opponent hands or future decks are exposed. This is a compact partial observation, not a complete history or a Markov-state guarantee.

Bids are 0–15 plus 25%, 50%, 75%, and 100% of the legal budget. Illegal and duplicate amounts are masked. Loan budgets include the loan amount. Reveals choose a color actually in hand, then map it to a valid index. The small bid grid deliberately cannot express every possible integer bid. Bid and reveal policies have separate weights. Missions preserve collected gems and rewards are counted at final scoring, matching this engine.

These are simple starting points: linear policies and terminal rewards limit expressiveness and make reveal credit assignment noisy. A neural policy, richer history, or DQN can be added behind the same bot interface later. Changing the feature layout/action grid requires retraining and a checkpoint-version bump.

## Checks

```sh
python3 -m unittest discover -s tests -v
```

Tests cover update direction, terminal MC targets, legal actions, mission tracking, actual-engine training, deterministic evaluation, and checkpoint round trips.
