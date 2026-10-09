"""Deprecated. Import :mod:`optimization_engine.presets` instead.

This module used to hold configuration persistence — save, load, rename a
named `EngineConfig`. It was never about a stress *scenario*, and the name
invited exactly that confusion right up until
:mod:`optimization_engine.stress` needed the word. So the module became
:mod:`optimization_engine.presets`, and this is what is left: every old name,
bound to the same object the new module exposes, plus one
:class:`DeprecationWarning` on import.

Nothing here changes behaviour. ``Scenario`` **is** ``Preset``, not a subclass
of it, so ``isinstance`` checks and pickles keep working, and the YAML and
JSON payloads are unchanged in both directions.

Scheduled for removal one release after 0.7.0. The mapping is mechanical:

===============================  =====================================
Old name                         New name
===============================  =====================================
``Scenario``                     :class:`~optimization_engine.presets.Preset`
``scenario_to_dict``             :func:`~optimization_engine.presets.preset_to_dict`
``scenario_from_dict``           :func:`~optimization_engine.presets.preset_from_dict`
``dump_scenarios_yaml``          :func:`~optimization_engine.presets.dump_presets_yaml`
``load_scenarios_yaml``          :func:`~optimization_engine.presets.load_presets_yaml`
``save_scenarios``               :func:`~optimization_engine.presets.save_presets`
``load_scenarios``               :func:`~optimization_engine.presets.load_presets`
``rename_scenario``              :func:`~optimization_engine.presets.rename_preset`
``delete_scenario``              :func:`~optimization_engine.presets.delete_preset`
``scenario_signature``           :func:`~optimization_engine.presets.preset_signature`
===============================  =====================================

``SCHEMA_VERSION``, ``NOTES_MAX_LEN``, ``config_signature`` and ``now_iso``
kept their names.
"""

from __future__ import annotations

import warnings

from optimization_engine.presets import (
    NOTES_MAX_LEN,
    SCHEMA_VERSION,
    Preset,
    Scenario,
    config_signature,
    delete_preset,
    delete_scenario,
    dump_presets_yaml,
    dump_scenarios_yaml,
    load_presets,
    load_presets_yaml,
    load_scenarios,
    load_scenarios_yaml,
    now_iso,
    preset_from_dict,
    preset_signature,
    preset_to_dict,
    rename_preset,
    rename_scenario,
    save_presets,
    save_scenarios,
    scenario_from_dict,
    scenario_signature,
    scenario_to_dict,
)

__all__ = [
    "NOTES_MAX_LEN",
    "Preset",
    "SCHEMA_VERSION",
    "Scenario",
    "config_signature",
    "delete_preset",
    "delete_scenario",
    "dump_presets_yaml",
    "dump_scenarios_yaml",
    "load_presets",
    "load_presets_yaml",
    "load_scenarios",
    "load_scenarios_yaml",
    "now_iso",
    "preset_from_dict",
    "preset_signature",
    "preset_to_dict",
    "rename_preset",
    "rename_scenario",
    "save_presets",
    "save_scenarios",
    "scenario_from_dict",
    "scenario_signature",
    "scenario_to_dict",
]

warnings.warn(
    "optimization_engine.scenarios has been renamed to "
    "optimization_engine.presets - it holds saved configurations, not stress "
    "scenarios, which are now in optimization_engine.stress. Every old name "
    "still resolves here (Scenario is Preset), and the file format is "
    "unchanged. This module is removed one release after 0.7.0.",
    DeprecationWarning,
    stacklevel=2,
)
