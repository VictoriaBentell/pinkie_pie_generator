import urllib
import urllib.request as request
import re
import os
import time

def collect_data(offset, sort):
	# Later delete images from My Little Perceptron
	os.chdir('/home/victoria/Machine_Learning/pinkie_pAI/original/unsorted')
	#offset = 0
	queue = 'pinkie+pie'
	# 16000 appears to be the maximum
	limit = 16000
	# newest/, popular-24-hours/, popular-all-time/
	#sort = 'popular-all-time/'
	cleanfiles = os.listdir('../pinkie/clean')
	while offset < limit:
		print('Offset: ' + str(offset))
		start = time.time()
		try:
			html_start = time.time()
			response = request.urlopen('https://www.deviantart.com/' + sort + '?q=' + queue + '&offset=' + str(offset))
			html = response.read()
			html = str(html)
			#print(html)
			quotes = [m.start() for m in re.finditer('"', html)]
			len_quotes = len(quotes)
			#print(len_quotes)
			i = 0
			j = 1
			quotearr = []
			for num in range(len_quotes - 1):
				new_quote = html[quotes[i] + 1:quotes[j]]
				len_quotearr = len(quotearr)
				duplicate = False
				if new_quote.find('img00') != -1:
					if new_quote.find('\\') == -1:
						quotearr.append(new_quote)
				i = i + 1
				j = j + 1
			#print(quotearr)
			html_end = time.time()
			#print('HTML time: ' + str(html_end - html_start))
			
			i = 0
			download_start = time.time()
			imagelist = []
			for image in quotearr:
				slashes = [m.start() for m in re.finditer('/', str(quotearr[i]))]
				full_name = str(quotearr[i])[slashes[-1] + 1:]
				extension = full_name[-4:]
				if extension == '.gif':
					print('Skipping ' + full_name)
					callexceptionnow = callexceptionnowplease
				name = str(quotearr[i])[slashes[-1] + 1:-12] + '.png'
				namejpg = str(quotearr[i])[slashes[-1] + 1:-12] + '.jpg'
				duplication = 0
				files = os.listdir('./')
				#print(str(image))
				# Try setting duplication as global
				for file in files:
					#print('Name: ' + name)
					#print('File: ' + file)
					if name == str(file):
						duplication = 1
						#print('Name: ' + name)
						#print('File: ' + file)
						
				for file in cleanfiles:
					if name == str(file):
						duplication = 1
					if namejpg == str(file):
						duplication = 1
				
				if duplication == 0:
					f = open(name,'wb')
					f.write(urllib.request.urlopen(image).read())
					f.close()
					imagelist.append(name)
				i = i + 1
			
			end = time.time()
			print('Total: %.4f  HTML: %.4f  Download: %.4f  Images: %d' % (end - start, html_end - html_start, end - download_start, len(imagelist)))
		except:
			print('HTTP ERROR: PAGE NOT FOUND')
		offset = offset + 7
	
def delete_unsorted():
	os.chdir('/home/victoria/Machine_Learning/pinkie_pAI/original/unsorted')
	for image in os.listdir('./'):
		os.remove(image)
		
#delete_unsorted()
#collect_data()
