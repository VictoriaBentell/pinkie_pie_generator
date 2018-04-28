import PIL
from PIL import Image
import os
import flip
# May want to do this resize for tensorflow object detection
#os.chdir('./data/pinkie/')

def add_to_dataset():
	print('removing...')
	for image in os.listdir('./data/pinkie/'):
		os.remove('./data/pinkie/' + image)
	
	basewidth = 64
	hsize = 64
	os.chdir('./original/pinkie/clean/')
	print('shrinking...')
	for image in os.listdir('./'):
		try:
			img = Image.open(image)
			img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
			#os.chdir('./resize/')
			img.save('../../../data/pinkie/' + image)
			#os.chdir('/home/victoria/Machine Learning/pinkie_pAI/data/pinkie/')
		except:
			print(str(image))
	print('flipping...')
	flip.flip()
	#print('rotating...')
	#flip.rotate()
	#print('adjusting brightness...')
	#flip.change_brightness()

