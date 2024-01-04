from networktables import NetworkTables
import cv2
import numpy as np
NetworkTables.initialize()
vid = cv2.VideoCapture(0)
table = NetworkTables.getTable("vision")
_, uhoh = vid.read()
data = cv2.imencode('.jpg', uhoh)[1].tostring()
while True:
    data = table.getRaw("Raw Frame", uhoh)
    nparr = np.frombuffer(data, np.uint8)
    newFrame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow("s", newFrame)
    print(table.getNumber("Target X",-3), table.getNumber("Target Y",-3))
    print(table.getValue("April Tags", -5))