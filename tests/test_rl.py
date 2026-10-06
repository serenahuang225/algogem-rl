import asyncio
import copy
from unittest.mock import AsyncMock
import tempfile
import unittest
from pathlib import Path
from algogem_client import AuctionCard, GemType, MissionCard
from algogem_rl import RLBot
from rl_policy import Policy
from train_rl import run


class RLTests(unittest.TestCase):
    def test_reinforce_increases_rewarded_probability(self):
        p = Policy()
        p.choose('x', [1.], [0, 1], True)
        chosen = p.trajectory[0][3]
        p.finish(1.)
        self.assertGreater(p.weights['x'][str(chosen)][0], p.weights['x'][str(1-chosen)][0])

    def test_mc_terminal_update(self):
        p = Policy('mc', learning_rate=0.5)
        p.choose('x', [1.], [0], True)
        p.finish(1.)
        self.assertAlmostEqual(p.weights['x']['0'][0], 0.5)

    def test_sdk_legality_and_mission_tracking(self):
        b = RLBot(Policy(), True)
        m = MissionCard(1, ['green'], 10, 'specific', 0)
        b.on_game_start(0, [GemType.GREEN], 4, 0, [], [m], {0: 0, 1: 4})
        for coins in (0, 1, 5, 100):
            b.coins = coins
            for kind in ('loan', 'invest', 'treasure_1', 'treasure_2'):
                card = AuctionCard(kind, 10 if kind in ('loan', 'invest') else None)
                for _ in range(10):
                    bid = b.get_bid(card, [])
                    self.assertTrue(0 <= bid <= coins + (10 if kind == 'loan' else 0))
        self.assertEqual(b.choose_card_to_reveal([GemType.GREEN]), 0)
        b.collection = [GemType.GREEN]
        b.on_mission_completed(0, m, [GemType.GREEN])
        self.assertEqual(b.collection, [GemType.GREEN])
        self.assertEqual(b.available_missions, [])
        self.assertEqual(b.mission_reward, 10)

    def test_network_protocol_training(self):
        from algogem_client import _AlgogemClient
        async def exercise(path):
            bot = RLBot(Policy(), True, path)
            client = _AlgogemClient(bot, 'localhost', 5000, 'test', 'RL', 'practice')
            client._send = AsyncMock()
            await client._handle_message(dict(type='game_start', player_id=0,
                hand=[{'gem_type': 'green'}], value_chart={'0': 0, '1': 4}))
            await client._handle_message(dict(type='bid_request',
                auction_card={'auction_type': 'treasure_1'},
                available_gems=[{'gem_type': 'green'}]))
            self.assertTrue(0 <= client._send.call_args.args[0]['amount'] <= 25)
            await client._handle_message(dict(type='reveal_request',
                hand=[{'gem_type': 'green'}]))
            self.assertEqual(client._send.call_args.args[0], {'type': 'reveal', 'card_index': 0})
            await client._handle_message(dict(type='game_end',
                final_scores={'0': 60, '1': 20, '2': 30, '3': 40}, winner=0))
            self.assertEqual(Policy.load(path).episodes, 1)
            self.assertTrue(bot.finished)
        with tempfile.TemporaryDirectory() as d:
            asyncio.run(exercise(Path(d)/'online.json'))

    def test_engine_training_save_reload_frozen_eval(self):
        for method in ('reinforce', 'mc'):
            p = Policy(method)
            run(p, 12, 123, True)
            self.assertEqual(p.episodes, 12)
            self.assertTrue(any(w for rows in p.weights.values() for row in rows.values() for w in row))
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'policy.json'
                p.save(path)
                loaded = Policy.load(path)
                before = copy.deepcopy(loaded.weights)
                first = run(loaded, 12, 500, False)
                second = run(Policy.load(path), 12, 500, False)
                self.assertEqual(first, second)
                # Evaluation must not change existing learned parameters.
                for h, rows in before.items():
                    for a, row in rows.items():
                        self.assertEqual(row, loaded.weights[h][a])
                self.assertEqual(loaded.episodes, 12)
                self.assertEqual(loaded.trajectory, [])

if __name__ == '__main__':
    unittest.main()
