from lxml.html import fromstring
import requests
from itertools import cycle
import traceback
import sys
import urllib.request, socket
from threading import Thread
class proxies:
	def __init__(self, n_proxies=50):
		self._n_proxies= n_proxies
		self._proxies= None
	def get_proxies(self):
		url = 'https://free-proxy-list.net/'
		response = requests.get(url)
		parser = fromstring(response.text)
		proxies = set()
		for i in parser.xpath('//tbody/tr'):
			if i.xpath('.//td[7][contains(text(),"yes")]'):
				proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
				proxies.add(proxy)
		return proxies
	def getproxy(self):
		if self._proxies==None:
			proxy_pool=cycle(self.get_proxies())
			self._proxies=1
		proxy_pool=next(proxy_pool)
		return proxy_pool
	
	def check_proxy(self,pip,teller):
		try:    
		   
			proxy_handler = urllib.request.ProxyHandler({'https': pip})        
			opener = urllib.request.build_opener(proxy_handler)
			opener.addheaders = [('User-agent', 'Mozilla/5.0')]
			urllib.request.install_opener(opener)        
			current_tile=str(teller)  # change the url address here
			
			urllib.request.urlretrieve('http://mt0.google.com/vt?lyrs=s&x='+str(133128+teller)+'&y=88009&z=18', current_tile)
			im = Image.open(current_tile)
			im.save(str(teller)+".jpg")
			print(teller)
			print(pip)
		except urllib.error.HTTPError as e:        
			return e
		except Exception as detail:
			return detail
		return 0
	
		
proxie=proxies()
socket.setdefaulttimeout(30)
proxe=proxie.get_proxies()

threads=[]
#prox=cycle(proxe)
for teller,proxy in enumerate(proxe):
	
	    thread = Thread( target=proxie.check_proxy, args=(proxy.strip(),teller,))
	    print(proxy.strip(),)
	    thread.start()
	    threads.append(thread)

for thread in threads:
	    thread.join()	
