"""Learned bidding and revealing, compatible with the existing TCP client."""
import argparse
import os
from collections import Counter
from algogem_client import AlgogemBot, run_bot
from rl_policy import Policy

COLORS = ('green', 'blue', 'pink', 'purple', 'orange')
KINDS = ('treasure_1', 'treasure_2', 'loan', 'invest')

def value(item):
    return getattr(item, 'value', item)


class RLBot(AlgogemBot):
    def __init__(self, policy, training=False, checkpoint=None):
        super().__init__(enable_logging=False)
        self.policy, self.training, self.checkpoint = policy, training, checkpoint
        self.finished = False
        self.errors = []

    def on_game_start(self, player_id, hand, num_players, starting_coins,
                      available_gems, available_missions, value_chart, **kwargs):
        super().on_game_start(player_id, hand, num_players, starting_coins,
                             available_gems, list(available_missions), value_chart, **kwargs)
        self.chart = value_chart.to_dict() if hasattr(value_chart, 'to_dict') else value_chart
        self.turn = self.debt = self.investments = self.mission_reward = 0
        self.policy.trajectory.clear()
        self.finished = False
        self.errors.clear()

    def features(self, card=None, gems=()):
        kind = value(card.auction_type) if card else ''
        x = [1., self.coins/100, self.turn/25, self.debt/100,
             self.investments/100, self.mission_reward/100, (card.value or 0)/25 if card else 0]
        x += [float(kind == k) for k in KINDS]
        x += [float(self.player_id == i) for i in range(4)]
        x += [self.player_coins.get(i, 0)/100 for i in range(4)]
        for items in (self.hand, self.collection, gems):
            counts = Counter(value(g) for g in items)
            x += [counts[c]/6 for c in COLORS]
        counts = {value(k): v for k, v in self.revealed_cards.items()}
        x += [counts.get(c, 0)/6 for c in COLORS]
        x += [self.chart.get(i, 0)/25 for i in range(7)]
        # Fixed slots retain individual mission requirements, including wildcards.
        for i in range(4):
            m = self.available_missions[i] if i < len(self.available_missions) else None
            counts = Counter(value(g) for g in m.required_gems) if m else {}
            x += [counts.get(c, 0)/4 for c in COLORS]
            x += [m.reward/25 if m else 0, m.wildcard_count/4 if m else 0]
            x += [float(m is not None and value(m.mission_type) == k)
                  for k in ('specific', 'n_same', 'n_different', 'two_pairs')]
        return x

    def get_bid(self, auction_card, available_gems):
        maximum = max(0, self.coins + ((auction_card.value or 0)
                      if value(auction_card.auction_type) == 'loan' else 0))
        # Exact low bids plus budget fractions for larger balances; mask duplicates.
        amounts = list(range(16)) + [round(maximum*f) for f in (.25, .5, .75, 1)]
        legal, seen = [], set()
        for a, bid in enumerate(amounts):
            if bid <= maximum and bid not in seen:
                legal.append(a)
                seen.add(bid)
        a = self.policy.choose('bid', self.features(auction_card, available_gems), legal, self.training)
        return amounts[a]

    def choose_card_to_reveal(self, hand):
        if not hand:
            return 0
        self.hand = list(hand)
        legal = [i for i, c in enumerate(COLORS) if any(value(g) == c for g in hand)]
        a = self.policy.choose('reveal', self.features(), legal, self.training)
        return next(i for i, g in enumerate(hand) if value(g) == COLORS[a])

    def on_auction_result(self, auction_card, winner, winning_bid, all_bids, gems_won):
        # Server engine uses enums; SDK uses strings.
        from algogem_client import AuctionCard
        card = AuctionCard(value(auction_card.auction_type), auction_card.value)
        super().on_auction_result(card, winner, winning_bid, all_bids, gems_won)
        self.turn += 1
        if winner == self.player_id:
            if card.auction_type == 'loan':
                self.debt += card.value
            elif card.auction_type == 'invest':
                self.investments += winning_bid + card.value

    def on_mission_completed(self, player_id, mission, gems_used):
        self.available_missions = [m for m in self.available_missions if m.mission_id != mission.mission_id]
        if player_id == self.player_id:
            self.mission_reward += mission.reward
        # Gems stay in collection; mission rewards are paid at final scoring.

    def on_game_end(self, final_scores, winner=None):
        if self.finished:
            return
        if self.training:
            # Optimize score margin over the mean opponent score, in units of 100.
            opponents = [s for p, s in final_scores.items() if p != self.player_id]
            self.policy.finish((final_scores[self.player_id] - sum(opponents)/len(opponents))/100)
            if self.checkpoint:
                self.policy.save(self.checkpoint)
        self.finished = True

    def on_rules_violation(self, *args):
        self.errors.append(args)

    def on_reject(self, reason, action, selected_value):
        raise RuntimeError(f'Server rejected {action}={selected_value}: {reason}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--server', default='localhost')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--token', default=os.environ.get('ALGOGEM_TOKEN'))
    parser.add_argument('--mode', choices=['practice', 'pvp'], default='practice')
    parser.add_argument('--train', action='store_true', help='Update and save after each complete practice game')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not args.token:
        parser.error('Provide --token or ALGOGEM_TOKEN')
    if args.train and args.mode != 'practice':
        parser.error('--train requires --mode practice')
    policy = Policy.load(args.model, args.seed)
    run_bot(RLBot(policy, args.train, args.model if args.train else None),
            server=args.server, port=args.port, token=args.token, mode=args.mode)

if __name__ == '__main__':
    main()
