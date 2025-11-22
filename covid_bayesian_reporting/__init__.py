"""
COVID-19 Bayesian Nowcasting and Data Imputation System.

A comprehensive statistical framework for addressing three critical challenges
in COVID-19 pandemic data reporting:

1. Severe Reporting Lags (14-21 days) → Bayesian Hierarchical Nowcasting
2. 30-40% Missing Race/Ethnicity Data → MICE with Census Tract Proxies
3. Inconsistent Lab Positivity Definitions → Bayesian Standardization
"""

__version__ = "1.0.0"
__author__ = "GenZ COVID-19 Response Team"

from .nowcasting import BayesianNowcaster
from .imputation import MICEImputer, CensusTractProxy
from .positivity_standardization import PositivityStandardizer
from .config import Config

__all__ = [
    'BayesianNowcaster',
    'MICEImputer',
    'CensusTractProxy',
    'PositivityStandardizer',
    'Config',
]
