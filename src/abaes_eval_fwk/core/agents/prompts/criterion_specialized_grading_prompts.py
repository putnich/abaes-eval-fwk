"""The default template for grading a single rubric criterion.

The wording itself lives in config_files/defaults/prompts.yaml, so a project can state its own
in prompts.yaml without the two drifting; this constant is what an agent built with no project
configuration falls back to.
"""

from abaes_eval_fwk.config.prompts_config import default_prompts_config

CRITERION_SPECIALIZED_GRADING_PROMPT: str = default_prompts_config().criterion_specialized
