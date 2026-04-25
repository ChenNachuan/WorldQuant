import random
import re
import logging
from typing import List, Dict, Tuple, Optional, Callable

logger = logging.getLogger(__name__)


class GeneticEngine:
    def __init__(
        self,
        operators: List[str] = None,
        fields: List[str] = None,
        windows: List[int] = None,
        groups: List[str] = None,
        population_size: int = 50,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ):
        self.operators = operators or [
            "ts_mean", "ts_std_dev", "ts_rank", "ts_sum",
            "rank", "zscore", "log", "sqrt",
            "divide", "subtract", "add", "multiply",
            "group_neutralize", "group_mean", "group_zscore",
        ]
        self.fields = fields or ["returns", "volume", "close", "open", "high", "low"]
        self.windows = windows or [5, 20, 60, 120, 252]
        self.groups = groups or ["sector", "industry"]
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.history: List[Dict] = []

    def evolve(
        self,
        population: List[Tuple[str, float]],
        fitness_fn: Optional[Callable] = None,
    ) -> List[str]:
        self.generation += 1
        logger.info(f"Generation {self.generation}: evolving {len(population)} individuals")

        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)

        elite_count = max(2, int(len(sorted_pop) * self.elite_ratio))
        elite = [expr for expr, _ in sorted_pop[:elite_count]]

        offspring = list(elite)

        while len(offspring) < self.population_size:
            r = random.random()

            if r < self.crossover_rate and len(sorted_pop) >= 2:
                parent1 = self._tournament_select(sorted_pop)
                parent2 = self._tournament_select(sorted_pop)
                children = self._crossover(parent1, parent2)
                offspring.extend(children)

            elif r < self.crossover_rate + self.mutation_rate:
                parent = self._tournament_select(sorted_pop)
                mutated = self._mutate(parent)
                offspring.append(mutated)

            else:
                expr = self._random_expression()
                offspring.append(expr)

        offspring = offspring[: self.population_size]

        self.history.append({
            "generation": self.generation,
            "best_fitness": sorted_pop[0][1] if sorted_pop else 0,
            "avg_fitness": sum(f for _, f in sorted_pop) / len(sorted_pop) if sorted_pop else 0,
            "population_size": len(offspring),
        })

        best_val = sorted_pop[0][1] if sorted_pop else 0
        logger.info(
            f"Generation {self.generation}: best={best_val:.4f}, "
            f"produced {len(offspring)} offspring"
        )
        return offspring

    def _tournament_select(
        self, population: List[Tuple[str, float]], tournament_size: int = 3
    ) -> str:
        tournament = random.sample(population, min(tournament_size, len(population)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def _crossover(self, parent1: str, parent2: str) -> List[str]:
        children = []

        parts1 = self._decompose(parent1)
        parts2 = self._decompose(parent2)

        if parts1 and parts2:
            cut1 = random.randint(1, max(1, len(parts1) - 1)) if len(parts1) > 1 else 0
            cut2 = random.randint(0, max(0, len(parts2) - 1)) if len(parts2) > 0 else 0

            child1_parts = parts1[:cut1] + parts2[cut2:]
            child2_parts = parts2[:cut2] + parts1[cut1:]

            child1 = self._recompose(child1_parts)
            child2 = self._recompose(child2_parts)

            if child1 and self._is_valid_expression(child1):
                children.append(child1)
            if child2 and self._is_valid_expression(child2):
                children.append(child2)

        if not children:
            children.append(self._mutate(parent1))
            children.append(self._mutate(parent2))

        return children

    def _mutate(self, expression: str) -> str:
        mutation_type = random.choice(["window", "field", "operator", "wrap", "unwrap"])

        if mutation_type == "window":
            numbers = re.findall(r"\b(\d+)\b", expression)
            if numbers and self.windows:
                old_num = random.choice(numbers)
                new_num = str(random.choice(self.windows))
                expression = expression.replace(old_num, new_num, 1)

        elif mutation_type == "field":
            tokens = re.findall(r"[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+", expression, re.IGNORECASE)
            non_op_tokens = [t for t in tokens if not any(t.startswith(p) for p in ["ts_", "group_"])]
            if non_op_tokens and self.fields:
                old_field = random.choice(non_op_tokens)
                new_field = random.choice(self.fields)
                expression = expression.replace(old_field, new_field, 1)

        elif mutation_type == "operator":
            ops_in_expr = re.findall(r"\b(ts_\w+|group_\w+|rank|zscore|log|sqrt)\b", expression)
            if ops_in_expr and self.operators:
                old_op = random.choice(ops_in_expr)
                candidates = [op for op in self.operators if op != old_op]
                if candidates:
                    new_op = random.choice(candidates)
                    expression = expression.replace(old_op, new_op, 1)

        elif mutation_type == "wrap":
            if self.fields:
                field = random.choice(self.fields)
                window = random.choice(self.windows)
                wrapper = random.choice(["rank", "zscore"])
                expression = f"{wrapper}({expression})"

        elif mutation_type == "unwrap":
            match = re.match(r"^(rank|zscore)\((.+)\)$", expression.strip())
            if match:
                expression = match.group(2)

        return expression

    def _random_expression(self) -> str:
        templates = [
            lambda: f"group_neutralize(zscore(ts_mean({random.choice(self.fields)}, {random.choice(self.windows)})), {random.choice(self.groups)})",
            lambda: f"rank(ts_std_dev({random.choice(self.fields)}, {random.choice(self.windows)}))",
            lambda: f"zscore(ts_delta({random.choice(self.fields)}, {random.choice(self.windows)}))",
            lambda: f"group_neutralize(zscore(rank(divide({random.choice(self.fields)}, {random.choice(self.fields)}))), {random.choice(self.groups)})",
            lambda: f"rank(divide(ts_mean({random.choice(self.fields)}, {random.choice(self.windows)}), ts_std_dev({random.choice(self.fields)}, {random.choice(self.windows)})))",
        ]
        return random.choice(templates)()

    def _decompose(self, expr: str) -> List[str]:
        parts = []
        depth = 0
        current = ""
        for ch in expr:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
                if depth == 0:
                    parts.append(current)
                    current = ""
            elif ch == "," and depth == 1:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        return parts

    def _recompose(self, parts: List[str]) -> str:
        if not parts:
            return ""
        return ", ".join(parts)

    def _is_valid_expression(self, expr: str) -> bool:
        if not expr or len(expr) < 5:
            return False
        balance = 0
        for ch in expr:
            if ch == "(":
                balance += 1
            elif ch == ")":
                balance -= 1
            if balance < 0:
                return False
        return balance == 0

    def get_generation_stats(self) -> Dict:
        if not self.history:
            return {"generation": 0, "best_fitness": 0, "avg_fitness": 0}
        last = self.history[-1]
        return {
            "generation": last["generation"],
            "best_fitness": round(last["best_fitness"], 4),
            "avg_fitness": round(last["avg_fitness"], 4),
            "total_generations": len(self.history),
        }
