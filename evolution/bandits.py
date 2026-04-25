import math
import random
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class Arm:
    def __init__(self, name: str):
        self.name = name
        self.pulls = 0
        self.rewards = 0.0
        self.reward_history: List[float] = []

    @property
    def average_reward(self) -> float:
        return self.rewards / self.pulls if self.pulls > 0 else 0.0

    def update(self, reward: float):
        self.pulls += 1
        self.rewards += reward
        self.reward_history.append(reward)


class MultiArmBandit:
    def __init__(
        self,
        arms: List[str] = None,
        strategy: str = "ucb1",
        exploration_constant: float = 2.0,
    ):
        self.strategy = strategy
        self.exploration_constant = exploration_constant
        self.arms: Dict[str, Arm] = {}
        self.total_pulls = 0

        if arms:
            for arm_name in arms:
                self.arms[arm_name] = Arm(arm_name)

    def add_arm(self, name: str):
        if name not in self.arms:
            self.arms[name] = Arm(name)

    def select(self) -> Optional[str]:
        if not self.arms:
            return None

        unexplored = [name for name, arm in self.arms.items() if arm.pulls == 0]
        if unexplored:
            return random.choice(unexplored)

        if self.strategy == "ucb1":
            return self._ucb1_select()
        elif self.strategy == "thompson":
            return self._thompson_select()
        elif self.strategy == "epsilon_greedy":
            return self._epsilon_greedy_select()
        else:
            return self._ucb1_select()

    def select_k(self, k: int) -> List[str]:
        selected = []
        temp_arms = dict(self.arms)

        for _ in range(min(k, len(temp_arms))):
            unexplored = [name for name, arm in temp_arms.items() if arm.pulls == 0]
            if unexplored:
                choice = random.choice(unexplored)
            else:
                if self.strategy == "ucb1":
                    choice = self._ucb1_select_from(temp_arms)
                else:
                    choice = self._ucb1_select_from(temp_arms)

            selected.append(choice)
            del temp_arms[choice]

        return selected

    def update(self, arm_name: str, reward: float):
        if arm_name not in self.arms:
            self.arms[arm_name] = Arm(arm_name)

        self.arms[arm_name].update(reward)
        self.total_pulls += 1

    def update_batch(self, results: List[Tuple[str, float]]):
        for arm_name, reward in results:
            self.update(arm_name, reward)

    def _ucb1_select(self) -> str:
        return self._ucb1_select_from(self.arms)

    def _ucb1_select_from(self, arms: Dict[str, Arm]) -> str:
        best_arm = None
        best_score = float("-inf")

        for name, arm in arms.items():
            if arm.pulls == 0:
                return name

            exploitation = arm.average_reward
            exploration = math.sqrt(
                self.exploration_constant * math.log(self.total_pulls) / arm.pulls
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_arm = name

        return best_arm or random.choice(list(arms.keys()))

    def _thompson_select(self) -> str:
        best_arm = None
        best_sample = float("-inf")

        for name, arm in self.arms.items():
            if arm.pulls == 0:
                sample = random.gauss(0, 1)
            else:
                mean = arm.average_reward
                std = 1.0 / math.sqrt(arm.pulls)
                sample = random.gauss(mean, std)

            if sample > best_sample:
                best_sample = sample
                best_arm = name

        return best_arm or random.choice(list(self.arms.keys()))

    def _epsilon_greedy_select(self, epsilon: float = 0.1) -> str:
        if random.random() < epsilon:
            return random.choice(list(self.arms.keys()))

        best_arm = None
        best_avg = float("-inf")
        for name, arm in self.arms.items():
            if arm.average_reward > best_avg:
                best_avg = arm.average_reward
                best_arm = name

        return best_arm or random.choice(list(self.arms.keys()))

    def get_top_arms(self, k: int = 10) -> List[Tuple[str, float, int]]:
        sorted_arms = sorted(
            self.arms.items(),
            key=lambda x: x[1].average_reward,
            reverse=True,
        )
        return [(name, arm.average_reward, arm.pulls) for name, arm in sorted_arms[:k]]

    def get_stats(self) -> Dict:
        total_pulls = sum(arm.pulls for arm in self.arms.values())
        total_reward = sum(arm.rewards for arm in self.arms.values())
        return {
            "strategy": self.strategy,
            "total_arms": len(self.arms),
            "total_pulls": total_pulls,
            "total_reward": round(total_reward, 4),
            "avg_reward": round(total_reward / total_pulls, 4) if total_pulls > 0 else 0,
            "top_arms": self.get_top_arms(5),
        }


class OperatorFieldBandit:
    def __init__(self, operators: List[str] = None, fields: List[str] = None):
        self.operator_bandit = MultiArmBandit(operators or [], strategy="ucb1")
        self.field_bandit = MultiArmBandit(fields or [], strategy="ucb1")
        self.combo_bandit = MultiArmBandit(strategy="ucb1")

    def select_combination(self, k_operators: int = 3, k_fields: int = 3) -> Dict:
        operators = self.operator_bandit.select_k(k_operators)
        fields = self.field_bandit.select_k(k_fields)
        return {"operators": operators, "fields": fields}

    def record_result(
        self,
        operators: List[str],
        fields: List[str],
        fitness: float,
    ):
        reward = min(fitness / 2.0, 1.0) if fitness > 0 else 0.0

        for op in operators:
            self.operator_bandit.add_arm(op)
            self.operator_bandit.update(op, reward)

        for field in fields:
            self.field_bandit.add_arm(field)
            self.field_bandit.update(field, reward)

        combo_key = f"{','.join(sorted(operators))}|{','.join(sorted(fields))}"
        self.combo_bandit.add_arm(combo_key)
        self.combo_bandit.update(combo_key, reward)

    def get_stats(self) -> Dict:
        return {
            "operators": self.operator_bandit.get_stats(),
            "fields": self.field_bandit.get_stats(),
            "combos": self.combo_bandit.get_stats(),
        }
