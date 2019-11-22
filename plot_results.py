import matplotlib.pyplot as plt
import csv
import os
import ast 

def read_results(source):
	for root, dirnames, filenames in os.walk(source):
		for i in filenames:
			j = 0
			confusion_matrixes = []
			validation_confusion_matrixes = []
			with  open(os.path.join(root,i),"r") as f:
				lines = f.readlines()
				final = []
				confusion_matrix = []
				validation_confusion_matrix = []
				for line in lines:
					if line[:19] == "validation accuracy":
						validation_accuracy = float(line.split(" : ")[1])
					elif line[:13] == "test accuracy":
						j+=1
						test_accuracy = float(line.split(" : ")[1])
						final.append((j,validation_accuracy,test_accuracy))
					elif line[:16] == "confusion_matrix":
						confusion_matrix = ast.literal_eval(line[17:])
						confusion_matrixes.append(confusion_matrix)
					elif line[:27] == "validation confusion_matrix":
						validation_confusion_matrix = ast.literal_eval(line[28:])
						validation_confusion_matrixes.append(validation_confusion_matrix)
				if len(final) > 0:
					with open(os.path.join("./csv_results/",i[:-4]+".csv"),"w") as output:
						for result in final:
							text = str(result[0]) + "," + str(result[1]) + "," + str(result[2]) +"\n"
							output.write(text)
					with open(os.path.join("./csv_results_confusion_matrix/",i[:-4]+".csv"),"w") as output:
						writer = csv.writer(output)
						y = 1
						for confusion_matrix in confusion_matrixes:
							x = 1
							writer.writerow([y]+range(1,7))
							for k in confusion_matrix:
								writer.writerow([x]+k)
								x+=1
							y+=1
							writer.writerow([" "," "," "," "," "," "," "])
					with open(os.path.join("./csv_results_valid_confusion_matrix/",i[:-4]+".csv"),"w") as output:
						writer = csv.writer(output)
						y = 1
						for validation_confusion_matrix in validation_confusion_matrixes:
							x = 1
							writer.writerow([y]+range(1,7))
							for k in validation_confusion_matrix:
								writer.writerow([x]+k)
								x+=1
							y+=1
							writer.writerow([" "," "," "," "," "," "," "])

def plot_files(source):
	for root, dirnames, filenames in os.walk(source):
		for i in filenames:
			x = []
			y = []
			z = []
			with open(os.path.join(root,i),'r') as csvfile:
				plots = csv.reader(csvfile, delimiter=',')
				for row in plots:
					x.append(int(row[0]))
					y.append(float(row[1]))
					z.append(float(row[2]))
			plt.plot(x,y,"r--",label='validation_accuracy')
			plt.plot(x,z,"b--",label='test_accuracy')
			plt.xlabel('epochs')
			plt.ylabel('accuracy')
			plt.title(i)
			plt.legend()
			figManager = plt.get_current_fig_manager()
			figManager.window.showMaximized()
			plt.savefig(os.path.join("./plots/",i[:-4]+'.png'))
			# plt.show()
			plt.close()
if __name__ == '__main__':
	# read_results("./results/")
	plot_files("./csv_results")