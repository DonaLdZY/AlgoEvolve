from itertools import combinations

import jsonschema
import pytest

from agents.planner.base_planner import build_closed_planning_json_schema, PREDICTION_PLANNING_MODULES


def test_every_selected_module_combination_is_representable_with_exact_original_output_shape():
    schema = build_closed_planning_json_schema(PREDICTION_PLANNING_MODULES)
    for size in (1, 2, 3):
        for modules in combinations(PREDICTION_PLANNING_MODULES, size):
            value = {"reason": "Use observed development evidence", "module": list(modules),
                     "plan": {name: "Keep fixed evaluator and make the stated change" for name in modules}}
            jsonschema.validate(value, schema)


def test_all_schema_objects_are_closed_and_require_every_declared_property():
    def inspect(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
                assert set(value["required"]) == set(value["properties"])
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(build_closed_planning_json_schema(PREDICTION_PLANNING_MODULES))


def test_unknown_or_empty_plan_map_is_rejected():
    schema = build_closed_planning_json_schema(PREDICTION_PLANNING_MODULES)
    for plan in ({}, {"unknown_module": "Do something"}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"reason": "test", "module": ["model_and_training"], "plan": plan}, schema)
