"""To be uploaded to the GoPiGo. This script starts a UDP server and listen for command messages
Once received, message is parsed to integers which are passed directly to the steer() method
to control each wheel's steering power, ranging from -100 to 100"""

import socket
from easygopigo3 import EasyGoPiGo3

GOPIGO_ADDR = '0.0.0.0'
GOPIGO_PORT = 55555

BUFFER = 1024


if __name__ == '__main__':
	# create UDP socket and bind to local port
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((GOPIGO_ADDR, GOPIGO_PORT))

	# EasyGoPiGo3 object to control the physical robots via API calls
	gpg = EasyGoPiGo3()
	while(True):
		msg = sock.recvfrom(BUFFER)[0].decode()
		if msg == "BYE":
			exit(0)
		l, r = map(int, msg.split())
		gpg.steer(l, r)
		