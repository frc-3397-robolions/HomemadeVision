from networktables import NetworkTables
NetworkTables.initialize(server="127.0.0.1")

table = NetworkTables.getTable('results')

while True:
    table.putNumber('data', 1)