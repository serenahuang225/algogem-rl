"""Train/evaluate against the existing server engine without network overhead."""
import argparse
import json
import random
import sys
from pathlib import Path
from algogem_rl import RLBot
from rl_policy import Policy

ENGINE = Path(__file__).resolve().parent / 'server/algogem_server_unpacked'
sys.path.insert(0, str(ENGINE))
from app.algogem.game import Game
from app.algogem.server.starter_bots import create_bot, get_all_bot_names


def run(policy, games, seed, training, opponents=None):
    names = opponents or get_all_bot_names()
    scores, wins, margins = [], 0, []
    for i in range(games):
        # Isolate policy RNG from opponent/global RNG. Rotate seats for tie fairness.
        random.seed(seed+i)
        bot = RLBot(policy, training)
        bots = [create_bot(random.choice(names)) for _ in range(3)]
        seat = i % 4
        bots.insert(seat, bot)
        result = Game(bots=bots, seed=seed+i).run_sync()
        if not bot.finished or bot.errors:
            raise RuntimeError(f'RL callbacks failed: finished={bot.finished}, errors={bot.errors}')
        score = result.final_scores[seat]
        scores.append(score)
        margins.append(score-sum(v for p, v in result.final_scores.items() if p != seat)/3)
        wins += result.winner == seat
    return dict(games=games, mean_score=sum(scores)/games, win_rate=wins/games,
                mean_margin=sum(margins)/games)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['reinforce', 'mc'], default='reinforce')
    p.add_argument('--games', type=int, default=2000)
    p.add_argument('--eval-games', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-seed', type=int, default=1000000)
    p.add_argument('--model', default='models/reinforce.json')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--opponents', nargs='+', choices=get_all_bot_names())
    p.add_argument('--learning-rate', type=float, default=0.01)
    args = p.parse_args()
    if args.games < 1 or args.eval_games < 1 or args.learning_rate <= 0:
        p.error('Games and learning rate must be positive')
    policy = Policy.load(args.model, args.seed) if args.resume or args.evaluate else Policy(
        args.method, args.seed, args.learning_rate)
    baseline = run(Policy(policy.method), args.eval_games, args.eval_seed, False, args.opponents)
    if not args.evaluate:
        remaining = args.games
        offset = policy.episodes
        while remaining:
            n = min(100, remaining)
            stats = run(policy, n, args.seed+offset, True, args.opponents)
            policy.save(args.model)
            print(json.dumps(dict(episodes=policy.episodes, training=stats)), flush=True)
            remaining -= n
            offset += n
    trained = run(policy, args.eval_games, args.eval_seed, False, args.opponents)
    report = dict(method=policy.method, episodes=policy.episodes,
                  evaluation_seed=args.eval_seed, opponents=args.opponents or 'all starter bots',
                  untrained=baseline, learned=trained)
    print(json.dumps(report, indent=2))
    Path(args.model).with_suffix('.evaluation.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
