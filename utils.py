import os
import re
# from mutagen.mp3 import MP3
# import wave
# import contextlib
# import scipy
# from scipy.io.wavfile import read
import numpy as np
# import pickle
import gc
import soundfile as sf

import pickle
import librosa

def extract_feature(source):
	for root, dirnames, filenames in os.walk(source):
		for i in filenames:
			final = np.array([])
			X, sample_rate = librosa.load(os.path.join(root,i))
			stft = np.abs(librosa.stft(X))
			mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
			chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
			mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
			contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
			tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
			final = np.concatenate((mfccs,chroma,mel,contrast,tonnetz),axis = 0)
			pickle_out = open(os.path.join(root,i.split('.')[0]+'.pkl'),"wb")
			pickle.dump(final, pickle_out)
			pickle_out.close()
			gc.collect()

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
 	return np.array(a[0],dtype=float)

def read_audio_file_data_pickle(source,interator,chunk_size):
	with open(source,"rb") as file_:
		data = pickle.load(file_)
	yield data[interator*chunk_size:interator*chunk_size+chunk_size]

def read_audio_file_data_pickle_test(source,chunk_size,number_of_chunks):
	with open(source,"rb") as file_:
		data = pickle.load(file_)
	for i in range(number_of_chunks):
		yield data[i*chunk_size:i*chunk_size+chunk_size]

def read_audio_file_data_chunks(source,interator,chunk_size,number_of_chunks):
	# for i in range(number_of_chunks):
	yield np.array(sf.read(start = interator*chunk_size,stop = interator*chunk_size+chunk_size,file = source)[0],dtype=float)

def read_audio_file_data_chunks_test(source,chunk_size,number_of_chunks):
	for i in range(number_of_chunks):
		yield np.array(sf.read(start = i*chunk_size,stop = i*chunk_size+chunk_size,file = source)[0],dtype=float)

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
	sf.write(destination,all_data,44100)
	return "success"
if __name__ == '__main__':
	extract_feature('./combined_wav_files/')
	# for i in read_audio_file_data_chunks("./combined_wav_files/ABA.wav",20,5):
		# print i
	# seperate_files("./recordings")
	# find_max_length("./l2arctic_release_v4.0/")
	# print len(read_audio_file_data("arctic_a0001.wav"))
	# for root,dirnames,filename in os.walk("./l2arctic_release_v4.0/"):
	# 	dir_names = dirnames
	# 	break
	# for i in dir_names:
	# 	print "Combining all wav files ","-------------",os.path.join("./l2arctic_release_v4.0/",i)
	# 	print combine_all_audio_files(os.path.join("./l2arctic_release_v4.0/",i,"wav"))
		# print "Deletion of ",os.path.join("./l2arctic_release_v4.0/",i,"main.npy")
		# try:
			# os.system("rm -f " + os.path.join("./l2arctic_release_v4.0/",i,"main.npy"))
			# print "success"
		# except:
			# print "failure"