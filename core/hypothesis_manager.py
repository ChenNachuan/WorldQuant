import os
import glob
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class HypothesisManager:
    """Manages alpha hypotheses extracted from forum posts or research.
    Inspired by AlphaSpire's Insight/Scraper logic.
    """
    def __init__(self, insights_dir: str = "insights", generator=None):
        self.insights_dir = insights_dir
        self.generator = generator
        self.theses: List[Dict] = []
        self._load_theses()

    def _load_theses(self):
        """Loads already processed theses from a cache file."""
        cache_path = os.path.join(self.insights_dir, "theses_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    self.theses = json.load(f)
                logger.info(f"Loaded {len(self.theses)} theses from cache.")
            except Exception as e:
                logger.error(f"Failed to load theses cache: {e}")

    def sync_insights(self):
        """Processes new files in the insights directory using the LLM to extract hypotheses."""
        if not self.generator:
            return

        # Find all .txt and .md files
        files = glob.glob(os.path.join(self.insights_dir, "*.txt")) + \
                glob.glob(os.path.join(self.insights_dir, "*.md"))
        
        processed_files = {t.get("source_file") for t in self.theses}
        
        new_theses = []
        for file_path in files:
            if file_path in processed_files:
                continue
                
            logger.info(f"Processing new insight: {os.path.basename(file_path)}")
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Use LLM to extract hypothesis
            thesis = self._extract_thesis(content)
            if thesis:
                thesis["source_file"] = file_path
                new_theses.append(thesis)
        
        if new_theses:
            self.theses.extend(new_theses)
            self._save_cache()
            logger.info(f"Extracted {len(new_theses)} new theses.")

    def _extract_thesis(self, content: str) -> Optional[Dict]:
        """Uses the generator's LLM to summarize a forum post into an alpha hypothesis."""
        prompt = f"""
You are a WorldQuant Alpha Expert. Analyze the following forum post/research snippet and extract a clear "Alpha Hypothesis".
The hypothesis should describe a market anomaly or signal logic that can be translated into a mathematical expression.

CONTENT:
{content}

OUTPUT FORMAT (JSON ONLY):
{{
    "title": "Short descriptive title",
    "hypothesis": "Detailed explanation of the signal logic",
    "tags": ["momentum", "reversal", "volume", etc.],
    "suggested_fields": ["field1", "field2"],
    "suggested_ops": ["op1", "op2"]
}}
"""
        try:
            # We assume generator has a direct prompt method or we use generate_alphas with a special mode
            # For now, let's assume we add a 'raw_prompt' method to generator
            response = self.generator.get_raw_completion(prompt)
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Hypothesis extraction failed: {e}")
        return None

    def _save_cache(self):
        cache_path = os.path.join(self.insights_dir, "theses_cache.json")
        with open(cache_path, 'w') as f:
            json.dump(self.theses, f, indent=2)

    def get_random_theses(self, count: int = 2) -> str:
        """Returns a string representation of random theses for prompt injection."""
        if not self.theses:
            return ""
        
        import random
        selected = random.sample(self.theses, min(count, len(self.theses)))
        lines = []
        for i, s in enumerate(selected, 1):
            lines.append(f"Hypothesis {i} [{s['title']}]: {s['hypothesis']} (Tags: {', '.join(s['tags'])})")
        return "\n".join(lines)
