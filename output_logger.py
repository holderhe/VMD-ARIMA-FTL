import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Logger(object):  	# 保存所有输出结果，避免各种出错导致重来
	def __init__(self, filename='default.txt', stream=sys.stdout, level=logging.DEBUG):
		self.terminal = stream
		self.log = open(filename, 'a+')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
		self.flush()  #每次写入后刷新到文件中，防止程序意外结束

	def flush(self):
		self.log.flush()
