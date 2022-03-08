"""GoPiGo remote controller. This script creates a UDP socket client to send command messages
to the GoPiGo whose UDP socket acts as the server. It then listens to keyboard events, arrow keys
press and release in particular, then transforms the current pressing/releasing keys into
a pair of values representing the steering values of GoPiGo's two wheels. Each time there is
a change in the pair's value, the new value is sent to the server. When user presses q to quit,
a '0 0' message is sent to command the robot to stop completely before exiting the program."""

import keyboard		# keyboard library, installed with 'pip3 install keyboard'
import socket

GOPIGO_ADDR = '192.168.1.34'
GOPIGO_PORT = 55555

KEYS = {'up', 'down', 'left', 'right'}
keypress = {
	k: False for k in KEYS
}


def sendMessage(sock, msg):
	sock.sendto(str.encode(msg), (GOPIGO_ADDR, GOPIGO_PORT))

def getSteeringValues():
	left, right = 100, 100
	k = keypress['up'] - keypress['down']	# up -> k=1, down -> k=-1, both equal -> k=0
	if keypress['left'] != keypress['right']:
		if keypress['left']:
			if k == 0:		# self-rotating to the left in the same spot
				return -100, 100
			left = 50
		else:
			if k == 0:		# self-rotating to the right in the same spot
				return 100, -100
			right = 50
	return k * left, k * right


if __name__ == '__main__':
	# create a simple UDP client socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	while True:
		event = keyboard.read_event()
		if event.name == 'q':
			sendMessage(sock, "0 0")	# command robot to a full stop before quitting
			sendMessage(sock, "BYE")
			exit(0)
		elif event.name not in KEYS:
			continue

		if (event.event_type == keyboard.KEY_DOWN and not keypress[event.name]) \
		or (event.event_type == keyboard.KEY_UP and keypress[event.name]):
			keypress[event.name] = not keypress[event.name]
			l, r = getSteeringValues()
			sendMessage(sock, f"{l} {r}")
