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
        """Intelligent crossover: swap sub-expressions between two parents.

        Strategy 1 (operator swap): Take outer operator from p1, inner args from p2
        Strategy 2 (field swap): Replace fields in p1 with fields from p2
        Strategy 3 (wrap swap): Wrap p2's core with p1's outer structure
        """
        children = []

        # Extract structure from both parents
        outer1, args1 = self._extract_outer_and_args(parent1)
        outer2, args2 = self._extract_outer_and_args(parent2)

        # Strategy 1: outer(p1) + args(p2) → child1; outer(p2) + args(p1) → child2
        if outer1 and args2:
            child1 = f"{outer1}({', '.join(args2)})"
            if self._is_valid_expression(child1):
                children.append(child1)
        if outer2 and args1:
            child2 = f"{outer2}({', '.join(args1)})"
            if self._is_valid_expression(child2):
                children.append(child2)

        # Strategy 2: swap fields between parents
        fields1 = re.findall(r"\b([a-z][a-z0-9_]*(?:_[a-z0-9_]+)+)\b", parent1, re.IGNORECASE)
        fields2 = re.findall(r"\b([a-z][a-z0-9_]*(?:_[a-z0-9_]+)+)\b", parent2, re.IGNORECASE)
        non_op1 = [f for f in fields1 if not any(f.startswith(p) for p in ["ts_", "group_"])]
        non_op2 = [f for f in fields2 if not any(f.startswith(p) for p in ["ts_", "group_"])]

        if non_op1 and non_op2:
            child3 = parent1.replace(random.choice(non_op1), random.choice(non_op2), 1)
            if child3 != parent1 and self._is_valid_expression(child3):
                children.append(child3)

        # Strategy 3: wrap one parent's core inside the other's outer function
        if outer1 and self._is_valid_expression(parent2):
            child4 = f"{outer1}({parent2})"
            if self._is_valid_expression(child4) and len(child4) < 300:
                children.append(child4)

        # Fallback: mutate both parents
        if not children:
            children.append(self._mutate(parent1))
            children.append(self._mutate(parent2))

        return children[:3]  # Return at most 3 children

    def _extract_outer_and_args(self, expr: str) -> Tuple[str, List[str]]:
        """Extract the outermost function call and its arguments.

        Example: 'rank(ts_corr(close, volume, 10))' → ('rank', ['ts_corr(close, volume, 10)'])
        Example: 'ts_mean(returns, 20)' → ('ts_mean', ['returns', '20'])
        """
        expr = expr.strip()
        match = re.match(r'^(-?\s*)([a-zA-Z_][a-zA-Z0-9_]*)\((.+)\)$', expr, re.DOTALL)
        if not match:
            return "", []

        prefix = match.group(1).strip()
        func_name = match.group(2)
        inner = match.group(3)
        outer = f"{prefix}{func_name}" if prefix else func_name

        # Split arguments respecting parentheses depth
        args = []
        depth = 0
        current = ""
        for ch in inner:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())

        return outer, args


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
