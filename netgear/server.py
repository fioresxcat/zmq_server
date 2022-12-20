# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

# define various tweak flags
options = {"flag": 0, "copy": False, "track": False}

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define Netgear server at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with client's IP address !!!
server = NetGear(
    address="192.168.50.96",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to server
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
