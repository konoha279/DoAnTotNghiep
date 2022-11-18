from core import *


exp = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label']

nonImg = NonImageToImage("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", exp)
nonImg.convert2Image()