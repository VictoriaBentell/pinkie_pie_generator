from __future__ import print_function
import os
from PIL import Image

def fusion():
	pinkie_dir = './original/pinkie/clean/transparency_new/'
	bg_dir = './original/pinkie/clean/transparency_new/Backgrounds/'
	bg_list = os.listdir(bg_dir)
	i = 0
	for image in os.listdir(pinkie_dir):
		try:
			foreground = Image.open(pinkie_dir + str(image))
			background = Image.open(bg_dir + str(bg_list[i]))
			w, h = foreground.size
	
			new = background.resize((w, h), Image.ANTIALIAS)
			
		
			new.paste(foreground, (0, 0), foreground)
			new.save(os.path.expanduser(pinkie_dir + 'output/' + str(image)))
			i = i + 1
			if i == len(bg_list):
				i = 0
		except:
			print(str(image) + ' is not a file')
		
		
		
fusion()