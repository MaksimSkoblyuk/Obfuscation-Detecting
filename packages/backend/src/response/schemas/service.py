import copy


def clean_schema(schema: dict) -> dict:
    cleaned_schema = copy.deepcopy(schema)
    if 'anyOf' in cleaned_schema:
        flag = False
        for pair in cleaned_schema['anyOf']:
            for key, value in pair.items():
                cleaned_schema[key] = value
                if key == "type" and value != 'null':
                    flag = True
            if flag:
                break
        del cleaned_schema['anyOf']
    for key, value in cleaned_schema.items():
        if isinstance(value, dict):
            cleaned_schema[key] = clean_schema(value)
    return cleaned_schema
