from PIL import Image
import os
from PIL import ImageEnhance
#im = np.fliplr(plt.imread(image))
def flip():
	os.chdir('/home/victoria/Machine_Learning/pinkie_pAI/')
	for image in os.listdir('/home/victoria/Machine_Learning/pinkie_pAI/data/pinkie/'):
		#try:
		im = Image.open('./data/pinkie/' + image)
		
		out = im.transpose(Image.FLIP_LEFT_RIGHT)
		out.save('./data/pinkie/flip' + image)
		#except: print('cant flip')
def rotate():
	os.chdir('/home/victoria/Machine_Learning/pinkie_pAI/')
	for image in os.listdir('/home/victoria/Machine_Learning/pinkie_pAI/data/pinkie/'):
		#try:
		arr = [90, 270]
		for rotation in arr:	
			
			im = Image.open('./data/pinkie/' + image)
			out = im.rotate(rotation)
			out.save('./data/pinkie/rotate-' + str(rotation) + '-deg-' + image)
		#except: print('cant flip')

def change_brightness():
	os.chdir('/home/victoria/Machine_Learning/pinkie_pAI/')
	for image in os.listdir('/home/victoria/Machine_Learning/pinkie_pAI/data/pinkie/'):
		i = 1
		try:
			while i <= 2:	
			
				im = Image.open('./data/pinkie/' + image)
				brightness = ImageEnhance.Brightness(im)
				enhancement = 1 - (i * .08)
				out = brightness.enhance(enhancement)
				out.save('./data/pinkie/brightness-dark-' + str(i) + '-' + image)
				i = i + 1
				
			i = 1
			while i <= 5:	
			
				im = Image.open('./data/pinkie/' + image)
				brightness = ImageEnhance.Brightness(im)
				enhancement = 1 + (i * .06)
				out = brightness.enhance(enhancement)
				out.save('./data/pinkie/brightness-bright-' + str(i) + '-' + image)
				i = i + 1

		except:
			print('Image has wrong code: ' + image)
