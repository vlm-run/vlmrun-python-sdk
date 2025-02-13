"""Pydantic utilities for VLMRun."""

from datetime import date, datetime, time, timedelta
from typing import Any, List, Type, Union, get_origin, Optional, Dict, Tuple

from pydantic import BaseModel, create_model, Field


def patch_response_format(response_format: Type[BaseModel]) -> Type[BaseModel]:
    """Patch the OpenAI response format to handle Pydantic models, including nested models.

    The following fields are not supported by OpenAI:
    - date
    - datetime
    - time
    - timedelta

    This function patches the response format to handle these fields. We convert them to strings and
    then convert them back to the original type.
    """

    def patch_pydantic_field_annotation(annotation: Any) -> Any:
        """Convert unsupported types to string."""
        if annotation in [date, datetime, time, timedelta]:
            return str
        elif get_origin(annotation) is Union:
            # For Union types, convert each argument
            args = annotation.__args__
            if not args:
                return str
            # Convert the first type
            first_type = patch_pydantic_field_annotation(args[0])
            # For single-type unions, combine with str
            if len(args) == 1:
                return Union[first_type, str]
            # For multi-type unions, use first two types
            second_type = patch_pydantic_field_annotation(args[1])
            return Union[first_type, second_type]
        elif get_origin(annotation) is List:
            return List[patch_pydantic_field_annotation(annotation.__args__[0])]
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return patch_pydantic_model(annotation)
        return annotation

    def patch_pydantic_model(model: Type[BaseModel]) -> Type[BaseModel]:
        """Create a new model with patched field types."""
        fields = model.model_fields.copy()
        new_fields = {
            field_name: (patch_pydantic_field_annotation(field.annotation), field)
            for field_name, field in fields.items()
        }
        return create_model(
            f"{model.__name__}_patched", __base__=BaseModel, **new_fields
        )

    return patch_pydantic_model(response_format)


def get_type_from_anyof(anyof_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    for type_dict in anyof_list:
        if type_dict.get("type") != "null":
            return (
                type_dict.get("type", ""),
                type_dict.get("format", ""),
                type_dict.get("$ref", ""),
            )
    return "", "", ""


def convert_type(schema_type: str, format: str = None, ref: str = None) -> str:
    """Convert JSON schema types to Python/Pydantic types."""
    type_mapping = {
        "string": {
            None: "str",
            "date": "date",
            "email": "EmailStr",
            "uri": "AnyUrl",
            "uuid": "UUID",
        },
        "number": {None: "float", "decimal": "Decimal"},
        "integer": "int",
        "boolean": "bool",
        "array": "List",
        "object": "dict",
    }

    if ref:
        return ref.split("/")[-1]

    if schema_type not in type_mapping:
        return "Any"

    if isinstance(type_mapping.get(schema_type), dict):
        return type_mapping[schema_type].get(format, type_mapping[schema_type][None])
    return type_mapping.get(schema_type, "Any")


def schema_to_pydantic_model(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """Convert a JSON schema to a Pydantic model class."""

    model_name = json_schema.get("title", "Response")

    nested_models = {}
    definitions = json_schema.get("$defs", {})
    for def_name, def_schema in definitions.items():
        nested_models[def_name] = schema_to_pydantic_model(def_schema)

    fields = {}
    properties = json_schema.get("properties", {})

    for prop_name, prop_schema in properties.items():
        if "anyOf" in prop_schema:
            schema_type, format, ref = get_type_from_anyof(prop_schema["anyOf"])

            if ref:
                ref_name = ref.split("/")[-1]
                field_type = Optional[nested_models[ref_name]]
            elif schema_type == "array":
                array_schema = next(
                    s for s in prop_schema["anyOf"] if s.get("type") == "array"
                )
                if "$ref" in array_schema["items"]:
                    item_type = nested_models[
                        array_schema["items"]["$ref"].split("/")[-1]
                    ]
                    field_type = Optional[List[item_type]]
                else:
                    item_type = convert_type(array_schema["items"].get("type"))
                    field_type = Optional[List[eval(item_type)]]
            else:
                python_type = convert_type(schema_type, format)
                field_type = Optional[eval(python_type)]
        else:
            python_type = convert_type(
                prop_schema.get("type"), prop_schema.get("format")
            )
            field_type = Optional[eval(python_type)]

        field = Field(default=None, description=prop_schema.get("description", ""))

        fields[prop_name] = (field_type, field)

    model = create_model(model_name, __base__=BaseModel, **fields)

    return model
