import logging
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    region: str
    universe: str
    delay: int = 1
    neutralization: str = "INDUSTRY"
    truncation: float = 0.01
    pasteurization: str = "ON"
    nan_handling: str = "OFF"
    unit_handling: str = "VERIFY"
    language: str = "FASTEXPR"
    instrument_type: str = "EQUITY"
    max_concurrent: int = 5
    decay: int = 0
    extra_settings: Dict = field(default_factory=dict)

    def to_simulation_settings(self) -> Dict:
        settings = {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": False,
        }
        settings.update(self.extra_settings)
        return settings


REGION_CONFIGS: Dict[str, RegionConfig] = {
    "USA": RegionConfig(
        region="USA",
        universe="TOP3000",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "USA_TOP500": RegionConfig(
        region="USA",
        universe="TOP500",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "USA_TOP200": RegionConfig(
        region="USA",
        universe="TOP200",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "CHN": RegionConfig(
        region="CHN",
        universe="TOP2000",
        neutralization="MARKET",
        max_concurrent=5,
    ),
    "EUR": RegionConfig(
        region="EUR",
        universe="TOP1200",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "ASI": RegionConfig(
        region="ASI",
        universe="TOP800",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "KOR": RegionConfig(
        region="KOR",
        universe="TOP500",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "TWN": RegionConfig(
        region="TWN",
        universe="TOP500",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
    "GLB": RegionConfig(
        region="GLB",
        universe="TOP3000",
        neutralization="INDUSTRY",
        max_concurrent=5,
    ),
}


def get_region_config(region_key: str) -> RegionConfig:
    if region_key in REGION_CONFIGS:
        return REGION_CONFIGS[region_key]
    for key, config in REGION_CONFIGS.items():
        if config.region == region_key:
            return config
    logger.warning(f"Unknown region '{region_key}', defaulting to USA")
    return REGION_CONFIGS["USA"]


def get_all_regions() -> List[str]:
    return sorted(set(c.region for c in REGION_CONFIGS.values()))


def get_all_universes() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for key, config in REGION_CONFIGS.items():
        region = config.region
        if region not in result:
            result[region] = []
        if config.universe not in result[region]:
            result[region].append(config.universe)
    return result
