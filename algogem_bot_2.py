"""ALGOGEM Bot - This is the file you edit to create your bot!

Implement your strategy by editing the MyBot class below.
Run with: python algogem_client_2.py --server <ip> --token <token> --mode practice

Data types available to you:
    GemType      - Enum: GREEN, BLUE, PINK, PURPLE, ORANGE
    AuctionCard  - Has auction_type ("treasure_1", "treasure_2", "loan", "invest")
                   and value (loan/invest amount, None for treasure)
    MissionCard  - Has mission_id, required_gems, reward, mission_type, wildcard_count

State tracked for you by the base class (access via self.*):
    coins               - Your current coin balance
    hand                - Your current hand (list of GemType)
    collection          - Gems you've won in auctions (list of GemType)
    player_coins        - All players' coin balances {player_id: coins}
    revealed_cards      - Cards in the value display {GemType: count}
    available_gems      - Gems currently available for auction (list of GemType)
    available_missions  - Mission cards still available this game
    completed_missions  - Missions you have completed this game
    player_names        - Names of all players [str, ...]
"""

from collections import Counter

from algogem_client_2 import AlgogemBot, AuctionCard, GemType, MissionCard  # noqa: F401


class MyBot(AlgogemBot):
    """Your bot - implement the two required methods below.

    The base class automatically tracks game state for you:
        self.coins              - Your current coin balance
        self.hand               - Your current hand (list of GemType)
        self.collection         - Gems you've won in auctions
        self.player_coins       - All players' coin balances {player_id: coins}
        self.revealed_cards     - Cards in the value display {GemType: count}
        self.available_gems     - Gems currently available for auction
        self.available_missions - Mission cards still available this game
        self.completed_missions - Missions you have completed this game
        self.player_names       - Names of all players [str, ...]
    """

    # --- REQUIRED: You must implement these two methods ---

    @staticmethod
    def _gem_key(gem: GemType | str) -> str:
        """Normalize GemType or string to a comparable gem key."""
        if isinstance(gem, GemType):
            return gem.value
        return str(gem)

    def _mission_helped_by_gems(
        self, mission: MissionCard, available_gems: list[GemType]
    ) -> bool:
        """Return True if any auction gem helps progress toward this mission."""
        auction_keys = {self._gem_key(g) for g in available_gems}
        if not auction_keys:
            return False

        if mission.mission_type == "specific":
            return bool(auction_keys & set(mission.required_gems))

        if mission.mission_type == "n_same":
            # Any gem of a color we don't yet have enough of helps.
            counts = Counter(self._gem_key(g) for g in self.collection)
            n = mission.wildcard_count
            return any(counts[self._gem_key(g)] < n for g in available_gems)

        if mission.mission_type == "n_different":
            have = {self._gem_key(g) for g in self.collection}
            return any(self._gem_key(g) not in have for g in available_gems)

        if mission.mission_type == "two_pairs":
            counts = Counter(self._gem_key(g) for g in self.collection)
            # A gem helps if it builds toward a pair we don't have yet.
            return any(counts[self._gem_key(g)] < 2 for g in available_gems)

        return False

    def _would_complete_mission(
        self, available_gems: list[GemType]
    ) -> MissionCard | None:
        """Return a mission that would be completed by winning these gems, if any."""
        gem_counts = Counter(
            self._gem_key(g) for g in self.collection + list(available_gems)
        )

        for mission in self.available_missions:
            if mission.mission_type == "specific":
                required = Counter(mission.required_gems)
                can_complete = all(
                    gem_counts[gem] >= need for gem, need in required.items()
                )
            elif mission.mission_type == "n_same":
                can_complete = any(
                    c >= mission.wildcard_count for c in gem_counts.values()
                )
            elif mission.mission_type == "n_different":
                can_complete = (
                    sum(1 for c in gem_counts.values() if c >= 1)
                    >= mission.wildcard_count
                )
            elif mission.mission_type == "two_pairs":
                can_complete = sum(1 for c in gem_counts.values() if c >= 2) >= 2
            else:
                can_complete = False

            if can_complete:
                return mission
        return None

    def _mission_bid_bonus(self, available_gems: list[GemType]) -> float:
        """Extra bid value from still-available missions for these auction gems."""
        if not available_gems or not self.available_missions:
            return 0.0

        bonus = 0.0
        for mission in self.available_missions:
            if not self._mission_helped_by_gems(mission, available_gems):
                continue
            helpful = sum(
                1 for g in available_gems if self._mission_helped_by_gems(mission, [g])
            )
            bonus += 0.5 * max(helpful, 1)
            bonus += 0.25 * mission.reward / 5

        completing = self._would_complete_mission(available_gems)
        if completing is not None:
            bonus += max(2, completing.reward // 4)

        return bonus

    def get_bid(
        self, auction_card: AuctionCard, available_gems: list[GemType]
    ) -> int:
        """Return your bid amount for this auction.

        This is called every auction. All 4 players bid simultaneously
        (sealed bid). Highest bidder wins; ties broken by seat order.

        Args:
            auction_card: The auction being bid on.
                .auction_type (str): What's being auctioned.
                    "treasure_1" - Winner pays bid, gets 1 gem from the pool
                    "treasure_2" - Winner pays bid, gets 2 gems from the pool
                    "loan"       - Winner pays bid, gets loan coins now,
                                   must repay loan value at game end
                    "invest"     - Winner pays bid, locks bid coins,
                                   gets bid + investment value at game end
                .value (int | None): The loan/investment amount.
                    None for treasure auctions.
                    For "loan": how many coins you receive now (and repay at end)
                    For "invest": bonus coins you receive at end on top of your bid

            available_gems: The face-up gem(s) you'd win (for treasure auctions).
                Each is a GemType enum value (e.g., GemType.GREEN).
                Empty list for loan/invest auctions.

        Returns:
            Your bid (0 or higher).
            - For treasure/invest: you can bid 0 up to your current coins
            - For loans: you can bid 0 up to coins + loan value
            - If your bid exceeds the allowed max, it defaults to 0
        """
        if auction_card.auction_type == "invest":
            return (auction_card.value or 0) // 2 + 2
        if auction_card.auction_type == "loan":
            return (auction_card.value or 0) // 3 + 1

        bid_amount = 5.0
        bid_amount += self._mission_bid_bonus(available_gems)

        if auction_card.auction_type == "treasure_2":
            bid_amount *= 1.7

        return min(int(bid_amount), self.coins)

    def choose_card_to_reveal(self, hand: list[GemType]) -> int:
        """Return which card to reveal from your hand (0 to len(hand)-1).

        After winning an auction, you must reveal one card from your hand
        to the "value display". The number of each gem type in the value
        display determines how much that gem type is worth at game end
        (see the value_chart passed in on_game_start).

        Args:
            hand: Your current hand of gem cards (list of GemType).

        Returns:
            Index of the card to reveal (0 to len(hand)-1).
        """
        return 0

    # --- OPTIONAL: Override any of these to hook into game events ---

    def on_game_start(
        self,
        player_id,
        hand,
        num_players,
        starting_coins,
        available_gems,
        available_missions,
        value_chart,
        round_number=None,
        game_number=None,
        games_per_round=None,
        player_names=None,
    ):
        super().on_game_start(
            player_id,
            hand,
            num_players,
            starting_coins,
            available_gems,
            available_missions,
            value_chart,
            round_number,
            game_number,
            games_per_round,
            player_names,
        )

    def on_mission_completed(self, player_id, mission, gems_used):
        super().on_mission_completed(player_id, mission, gems_used)
