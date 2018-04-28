import os

for filename in os.listdir('./original/unsorted/'):
	for cleanfilename in os.listdir('./original/pinkie/clean/'):
		if filename == cleanfilename:
			print(filename)
			# REMEMBER TO CHANGE THE EXTENSIONS TO .PNG?????			
			#os.remove('./original/drive photos/' + filename)
