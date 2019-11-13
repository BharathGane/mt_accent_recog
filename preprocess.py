import librosa
import pickle
import os 
import numpy as np
import gc
def pkl_dump(source):
	for root, dirnames, filenames in os.walk(source):
		for i in filenames:
			data,sample_rate = librosa.load(os.path.join(root,i),sr=44100)
			pickle_out = open(os.path.join("./pkl_files",i.split('.')[0]+'.wav'),"wb")
			pickle.dump(data, pickle_out)
			pickle_out.close()
			gc.collect()


def extract_features(source):
	for root, dirnames, filenames in os.walk(source):
		for i in filenames:
			print i
			final = []
			with open(os.path.join(root,i),"rb") as file_:
				data = pickle.load(file_)
			sample_rate = 44100
			stft = np.abs(librosa.stft(data))
			mfccs = np.mean(np.transpose(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)),axis=1)
			chroma = np.mean(np.transpose(librosa.feature.chroma_stft(S=stft, sr=sample_rate)),axis=1)
			mel = np.mean(np.transpose(librosa.feature.melspectrogram(data, sr=sample_rate)),axis=1)
			contrast = np.mean(np.transpose(librosa.feature.spectral_contrast(S=stft, sr=sample_rate)),axis=1)
			final = np.transpose(np.vstack((mfccs,chroma,mel,contrast)))
			pickle_out = open(os.path.join("./final_features2",i),"wb")
			pickle.dump(final, pickle_out)
			pickle_out.close()
			gc.collect()



if __name__ == '__main__':
	extract_features("./pkl_files")