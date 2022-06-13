import pandas as pd 
import matplotlib.pyplot as plt 


data=[["FL32",0.4758,0.6456],
      ["OL",0.7767,0.7846],
      ["GVision",0.516,0.650],
      ["FL32_GVision",0.681,0.771],
      ["OL_GVision",0.825,0.869]
     ]

df=pd.DataFrame(data,columns=["Datasets","YOLOv4","YOLOv5"])
df.plot(x="Datasets", y=["YOLOv4","YOLOv5"], kind="bar",figsize=(9,7))
plt.tight_layout()
plt.show()

data=[["Cocacola",0.639,0.782, 0.662, 0.742, 0.792],
      ["Fedex",0.772, 0.947, 0.961, 0.962, 0.967],
      ["Ford",0.487, 0.896, 0.661, 0.695, 0.896],
      ["Pepsi",0.678, 0.742, 0.244, 0.641, 0.804],
      ["Shell",0.652,0.847,0.724,0.815,0.886]
     ]

df=pd.DataFrame(data,columns=["Datasets","FL32","OL", "GVision", "FL32_GVision", "OL_GVision"])
df.plot(x="Datasets", y=["FL32","OL", "GVision", "FL32_GVision", "OL_GVision"],figsize=(9,7))
plt.tight_layout()
plt.show()

data=[["Cocacola",0.639, 0.662],
      ["Fedex",0.772, 0.961],
      ["Ford",0.487, 0.661],
      ["Pepsi",0.678, 0.244],
      ["Shell",0.652, 0.724]
     ]

font = {'family' : 'normal',
        'size'   : 14,
        'weight' : 'bold'}

plt.rc('font', **font)

df=pd.DataFrame(data,columns=["Classes","FlickrLogos32","Google Vision"])
df.plot(x="Classes", y=["FlickrLogos32","Google Vision"], kind="bar",figsize=(9,7))
plt.title('AP for each class')
plt.ylabel('AP')
plt.tight_layout()
plt.show()