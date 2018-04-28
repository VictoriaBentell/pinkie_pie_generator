import os

for filename in os.listdir('./original/pinkie/clean/'):
	if str(filename).find('-') != -1:
		os.rename('./original/pinkie/clean/' + filename, './original/pinkie/clean/' + filename[:-12] + '.png')