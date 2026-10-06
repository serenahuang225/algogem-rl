"""Small, generic episodic RL policies using only Python's standard library."""
import json
import math
import random
from pathlib import Path


class Policy:
    def __init__(self, method='reinforce', seed=0, learning_rate=0.01, epsilon=0.15):
        if method not in ('reinforce', 'mc'):
            raise ValueError('method must be reinforce or mc')
        self.method, self.learning_rate, self.epsilon = method, learning_rate, epsilon
        self.rng = random.Random(seed)
        self.weights = {}
        self.episodes = 0
        self.baseline = 0.0
        self.trajectory = []

    def choose(self, head, features, legal, training=False):
        """features: state vector; legal: allowed discrete action IDs."""
        if not legal:
            raise ValueError('No legal actions')
        rows = self.weights.setdefault(head, {})
        for a in legal:
            rows.setdefault(str(a), [0.0] * len(features))
            if len(rows[str(a)]) != len(features):
                raise ValueError('Checkpoint feature dimensions do not match')
        values = [sum(w*x for w, x in zip(rows[str(a)], features)) for a in legal]
        exps = [math.exp(v-max(values)) for v in values]
        probs = [v/sum(exps) for v in exps]
        if training and self.method == 'reinforce':
            action = self.rng.choices(legal, weights=probs)[0]
        elif training and self.rng.random() < self.epsilon:
            action = self.rng.choice(legal)
        else:
            action = legal[values.index(max(values))]
        if training:
            self.trajectory.append((head, list(features), list(legal), action, probs))
        return action

    def finish(self, reward):
        """Undiscounted terminal return; update only after a complete episode."""
        advantage = reward - self.baseline
        for head, x, legal, chosen, probs in self.trajectory:
            rows = self.weights[head]
            if self.method == 'reinforce':
                for a, p in zip(legal, probs):
                    delta = self.learning_rate * advantage * ((a == chosen) - p)
                    rows[str(a)] = [w + delta*f for w, f in zip(rows[str(a)], x)]
            else:
                row = rows[str(chosen)]
                error = reward - sum(w*f for w, f in zip(row, x))
                # Normalized SGD prevents large feature vectors exploding.
                step = self.learning_rate * error / max(1.0, sum(f*f for f in x))
                rows[str(chosen)] = [w + step*f for w, f in zip(row, x)]
        self.baseline += 0.05 * advantage
        self.episodes += 1
        self.trajectory.clear()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(version=1, method=self.method, learning_rate=self.learning_rate,
                    epsilon=self.epsilon, episodes=self.episodes,
                    baseline=self.baseline, weights=self.weights)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(data, allow_nan=False))
        temporary.replace(path)

    @classmethod
    def load(cls, path, seed=0):
        data = json.loads(Path(path).read_text())
        if data['version'] != 1:
            raise ValueError('Unsupported policy version')
        policy = cls(data['method'], seed, data['learning_rate'], data['epsilon'])
        policy.weights = data['weights']
        policy.episodes, policy.baseline = data['episodes'], data['baseline']
        return policy
