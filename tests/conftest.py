"""Test-session settings for deterministic local execution."""

import os


os.environ["LOKY_MAX_CPU_COUNT"] = "1"
