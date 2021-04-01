# author:caijiawei
# time: 2020.3.10
"""
this is used to start, stop. Therefore, we need to listen to the two command. So this file virtually is sending start
and stop commend. we will use broadcast communication. When devices are in different local network, we can use multicast
 instead.Please pay attention, we use 22223 as command receiving port.
"""
import socket
import json
import time


class SystemCommand:
	start = 'start'  # when launch all devices, we should send the launch messsage after.
	stop = 'stop'


class SendCommand:
	HOST = '<broadcast>'
	PORT = 22223  # receiver's port
	BUFSIZE = 1024
	ADDR = (HOST, PORT)  # broadcast address

	def start_system(self, key):
		self.send_message(SystemCommand.start)  # send start message
		self._before_start_reformat_the_key(key)
		time.sleep(2)
		self.send_message(key)
		print('send system starting message!')

	def stop_system(self):
		self.send_message(SystemCommand.stop)
		print('send system stopping message!')

	def send_message(self, file):
		udpCliSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		udpCliSock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
		if isinstance(file, str):
			udpCliSock.sendto(file.encode(), self.ADDR)
		elif isinstance(file, dict):
			json_string = json.dumps(file).encode()
			udpCliSock.sendto(json_string, self.ADDR)
		udpCliSock.close()

	@staticmethod
	def _before_start_reformat_the_key(key):
		"""
		this function is used to reformat the device order.
		1.rank all devices by there computing capability.
		2.get the server device.
		:param key:ring_key is a list
		:return:None
		"""
		key['device'].sort(key=lambda x: x[1])  #
		key['server'] = []
		key['server'] = [[buf[0]] for buf in key['device'][-1*key['PS_number']:len(key['device'])]]  # take the worst computing device as server. Here is ip address.

	@staticmethod
	def test_del():
		print(SystemCommand.start, 'amazing! Its good! Brilliant!')


if __name__ == '__main__':
	key = {'this is ring_key': 'let us check it out!'}
	SendCommand().start_system(key)
