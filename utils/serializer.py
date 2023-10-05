from json import loads, dumps

# Define value deserializer function
def json_deserializer(val):
    return loads(val.decode('utf-8'))

# Define value serializer function
def json_serializer(val):
    return dumps(val).encode('utf-8')