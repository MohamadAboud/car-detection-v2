from json import loads

# Define value deserializer function
def json_deserializer(val):
    return loads(val.decode('utf-8'))