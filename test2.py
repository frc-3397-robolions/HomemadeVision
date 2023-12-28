from networktables import NetworkTables
NetworkTables.initialize()

table = NetworkTables.getTable("vision")

while True:
    print(table.getNumber("Target X",-3), table.getNumber("Target Y",-3))