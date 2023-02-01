import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Logger(object):  	# ����������������������ֳ���������
	def __init__(self, filename='default.txt', stream=sys.stdout, level=logging.DEBUG):
		self.terminal = stream
		self.log = open(filename, 'a+')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
		self.flush()  #ÿ��д���ˢ�µ��ļ��У���ֹ�����������

	def flush(self):
		self.log.flush()
