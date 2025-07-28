import os
import sys
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def clean_plt(ax):
    ax.tick_params(direction='out', length=2, width=1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(0)
    ax.tick_params(labelsize=10.0)
    ax.tick_params(axis='x', which='major', pad=2.0)
    ax.tick_params(axis='y', which='major', pad=2.0)
    return ax


def log_subsection(title: str):
    border = "#" * 60
    logger.info(f"\n{border}")
    logger.info(f"### {title.center(52)} ###")
    logger.info(f"{border}\n")


def log_section(title: str):
    border = "#" * 60
    logger.info(f"\n{border}")
    logger.info(f"### {title.upper().center(52)} ###")
    logger.info(f"{border}\n")

