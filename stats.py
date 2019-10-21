import os
import re
from mutagen.mp3 import MP3
import wave
import contextlib
import scipy
from scipy.io.wavfile import read
import numpy as np
import pickle
import gc
import soundfile as sf

def seperate_files(source):
	max_length = 0
	max_length_np = 0
	temp = 0
	for root, dirnames, filenames in os.walk(source):
		file_names = []
		# temp+=len(filenames)
		for i in filenames:
			x = re.split('(\d+)',i)
			if x[0] not in file_names:
				file_names.append(x[0])
				os.system("mkdir .temp//"+x[0])
			os.system("cp "+root+"/"+i+" ./temp/" +x[0]+"/")

def find_max_length(source):
	max_length = 0
	max_length_np = 0
	temp = 0
	for root, dirnames, filenames in os.walk(source):
		dir_names = dirnames
		break
	for i in dir_names:
		accent_length = 0
		for root, dirnames,filenames in os.walk(os.path.join(source,i,"wav")):
			for audio_file in filenames:
				with contextlib.closing(wave.open(os.path.join(root,audio_file),'r')) as f:
					frames = f.getnframes()
					rate = f.getframerate()
					duration = frames / float(rate)
					accent_length += duration
		# if accent_length>=2000:
		print(str(i)+","+str(accent_length))

def read_audio_file_data(source):
	a = sf.read(source)
	print a[1]
	return np.array(a[0],dtype=float)
	gc.collect()

def dump_pickle(data,file_path):
	file = open(file_path, 'wb')
	pickle.dump(data, file)
	file.close()

def combine_all_audio_files(source):
	all_data = np.array([])
	for root, dirnames, filenames in os.walk(source):
		for i in filenames:
			gc.collect()
			if i[0] == ".":
				pass
			else:
				all_data = np.concatenate((all_data,read_audio_file_data(os.path.join(root,i))),axis = 0)
	np.save(os.path.join(source,"main.npy"),all_data)
	return "success"

def np_to_wav(source,destination):
	all_data = np.load(source)
	sf.write(destination,all_data,44100)
	return "success"
if __name__ == '__main__':
	# seperate_files("./recordings")
	# find_max_length("./l2arctic_release_v4.0/")
	# print len(read_audio_file_data("arctic_a0001.wav"))
	for root,dirnames,filename in os.walk("./l2arctic_release_v4.0/"):
		dir_names = dirnames
		break
	for i in dir_names:
		print "Conversion from wav files to combined npy file ","-------------",os.path.join("./l2arctic_release_v4.0/",i)
		print combine_all_audio_files(os.path.join("./l2arctic_release_v4.0/",i,"wav"))
		print "Conversion from npy","------------------",os.path.join("./l2arctic_release_v4.0/",i)
		print np_to_wav(os.path.join("./l2arctic_release_v4.0/",i,"main.npy"),os.path.join("./l2arctic_release_v4.0/",i,"final.wav")
		print "Deletion of ",os.path.join("./l2arctic_release_v4.0/",i,"main.npy")
		try:
			os.system("rm -f " + os.path.join("./l2arctic_release_v4.0/",i,"main.npy"))
			print "success"
		except:
			print "failure"