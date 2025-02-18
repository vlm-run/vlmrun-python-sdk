from typing import Type, Dict, Any
from pydantic import BaseModel, create_model
import graphql
from loguru import logger


def create_pydantic_model_from_gql(
    base_model: Type[BaseModel], gql_query: str
) -> Type[BaseModel]:
    """Creates a subset Pydantic model based on a GraphQL query.

    Args:
        base_model: The original Pydantic model containing all fields
        gql_query: GraphQL query string specifying desired fields

    Returns:
        A new Pydantic model containing only the fields specified in the query

    Raises:
        graphql.GraphQLSyntaxError: If the GQL query is invalid
        ValueError: If the query references fields not in the base model
    """
    try:
        ast = graphql.parse(gql_query)
    except graphql.GraphQLSyntaxError as e:
        logger.error(f"Invalid GraphQL query: {e}")
        raise

    fields: Dict[str, Any] = {}

    def extract_fields(node, current_fields):
        if isinstance(node, graphql.SelectionSet):
            for selection in node.selections:
                if isinstance(selection, graphql.Field):
                    field_name = selection.name.value
                    original_field = base_model.model_fields.get(field_name)

                    if original_field is None:
                        continue

                    if selection.selection_set:
                        nested_fields = {}
                        extract_fields(selection.selection_set, nested_fields)
                        if nested_fields:
                            current_fields[field_name] = (
                                create_model(
                                    f"{base_model.__name__}{field_name.capitalize()}",
                                    **nested_fields,
                                ),
                                original_field.default,
                            )
                    else:
                        current_fields[field_name] = (
                            original_field.annotation,
                            original_field.default,
                        )

    for definition in ast.definitions:
        if isinstance(definition, graphql.OperationDefinition):
            extract_fields(definition.selection_set, fields)

    if not fields:
        raise ValueError("No valid fields found in GraphQL query")

    return create_model(f"{base_model.__name__}", __base__=BaseModel, **fields)
