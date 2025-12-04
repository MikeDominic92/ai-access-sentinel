"""
CrowdStrike Falcon ITDR Integration Module.

v1.1 Enhancement - December 2025

This module provides integration with CrowdStrike Falcon for:
- Identity Threat Detection and Response (ITDR)
- Threat intelligence enrichment
- Alert correlation with ML anomaly detection
- Real-time identity protection events
"""

from .crowdstrike_connector import CrowdStrikeConnector
from .falcon_event_parser import FalconEventParser
from .alert_correlator import AlertCorrelator

__all__ = [
    'CrowdStrikeConnector',
    'FalconEventParser',
    'AlertCorrelator'
]

__version__ = "1.1.0"
