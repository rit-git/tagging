import numpy as np
#import pandas as pd
import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt

names = "LR    SVM    CNN    LSTM    BERT".split("    ")
colors = ['tab:gray', 'tab:green','tab:orange',  'tab:red', 'tab:blue', 'orange' ]
patterns = ('--', '\\', '////', '\\\\', '\\\\', '\\\\', '.', '*')
fills = [False, True, False, False, True, True, False]

#colors = ['xkcd:light red', 'xkcd:dark red','xkcd:bright blue',  'xkcd:blue', 'xkcd:dodger blue', 'orange']
#colors = ['xkcd:red', 'xkcd:red','xkcd:blue',  'xkcd:blue', 'xkcd:blue', 'orange']
#patterns = ('--', '\\', '////', '\\\\', '\\\\', '\\\\', '.', '*')
#fills = [True, True, True, True, True, True, True]
#fills = [False, False, False, False, False, False, False]

#colors = ['tab:blue', 'tab:blue','tab:blue',  'tab:blue', 'tab:blue', 'blue' ]
#patterns = ('--', '++', 'x', '*', '.')
#fills = [False, False, False, False, False, False, False]

font = {'family' : 'normal',
       'weight' : 'normal',
       'size'   : 16}
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}

group1 = [["HOTEL", "SENT", "PARA", "REQ", "REF", "QUOTE", "SUPPORT", "AGAINST"], ["FUNNY-", "BOOK-"]]
group2 = [["SUGG", "HOMO", "HETER", "TV", "EVAL", "FACT", "ARGUE"], ["AMAZON-", "YELP", "FUNNY*", "BOOK*"]]

def plot(ds, datasets): 
  plt.rcParams.update(params)
  plt.figure(figsize=(6,4))

  matplotlib.rc('font', **font)  
  bar_width = 0.7

  for i in range(len(names)):
    plt.bar(i, datasets[ds][i], bar_width, color=colors[i], 
      edgecolor=colors[i], hatch=patterns[i], fill = fills[i],
      label=names[i])
    #bars = plt.bar(names, datasets[ds])
    #plt.ylim([0.0, 1.0])
    #for bar, pattern, color in zip(bars, patterns, colors):
    #   bar.set_color(color)
       # bar.set_hatch(pattern)
  plt.title(ds)
  plt.xticks([i for i in range(len(names))], names)
  plt.ylabel('AUC')
  plt.tight_layout()
  plt.savefig('%s.pdf' % ds, format='pdf')

def plot_all(datasets):
  plt.rcParams.update(params)
  plt.figure(figsize=(24,10))
  matplotlib.rc('font', **font)  
  bar_width = 0.14
  bar_break = 0.02
  
  xaxis = [i for i in range(len(datasets.keys()))]
  xticks = list(datasets.keys())
  for i in range(len(xticks)):
    for j in range(len(names)):
      plt.bar(i+(j-2.5)*(bar_width+bar_break)+bar_width/2, 
        datasets[xticks[i]][j], bar_width, color=colors[j], 
        edgecolor=colors[j], hatch=patterns[j], fill = fills[j],
        label=names[j])

  plt.xticks(xaxis, xticks, rotation=20)
  plt.ylabel('AUC')
  plt.tight_layout()
  plt.savefig('all.pdf', format='pdf')

def plot_per_group(groups, datasets, name):
  plt.rcParams.update(params)
  plt.figure(figsize=(20,6))
  matplotlib.rc('font', **font)  
  bar_width = 0.14
  bar_break = 0.02
  
  size = sum([len(g) for g in groups])+len(groups)-1
  xaxis = [i for i in range(13)]
  xticks = []
  for g in groups:
    xticks += g + [""]
  while len(xticks) < len(xaxis):
    xticks.append("")
  print(len(xticks))
  for i in range(len(xticks)):
    if len(xticks[i]) < 1:
      continue
    for j in range(len(names)):
      if i == 0:
        plt.bar(i+(j-2.5)*(bar_width+bar_break)+bar_width/2, 
          datasets[xticks[i]][j], bar_width, color=colors[j], 
          edgecolor=colors[j], hatch=patterns[j], fill = fills[j],
          label=names[j])
      else:
        plt.bar(i+(j-2.5)*(bar_width+bar_break)+bar_width/2, 
          datasets[xticks[i]][j], bar_width, color=colors[j], 
          edgecolor=colors[j], hatch=patterns[j], fill = fills[j])
  # TODO: finish the annotation.
  annotations = ["Small", "Large"]
  for i in range(len(groups)):
    start = len(groups[i-1])+0.5 if i > 0 else -0.5
    end = start + len(groups[i])
    plt.annotate("", xy=(start, -0.14), xytext=(end, -0.14),
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
             annotation_clip=False)
    plt.annotate(annotations[i], xy = ((end+start-(len(annotations[i])/3))/2, -0.15),
     xytext=((end+start-(len(annotations[i])/10))/2, -0.15), bbox=dict(boxstyle="round", edgecolor="w", fc="w"),
     annotation_clip=False)

  plt.ylim([0.0, 1.0])
  plt.xticks(xaxis, xticks, rotation=20)
  plt.ylabel('AUC')
  plt.legend(ncol=len(names), loc='upper center', 
    fancybox="False", mode="expand", framealpha=0.0, bbox_to_anchor=(0, 1.01, 1, 0.1))

  plt.tight_layout()
  plt.savefig(name, format='pdf')

# F1 score
#data = """Dataset    LR    SVM    CNN    LSTM    BERT
#SUGG    0.79    0.77    0.77    0.68    0.86
#HOTEL    0.53    0.55    0.46    0.59    0.67
#SENT    0.5    0.51    0.43    0.45    0.57
#PARA    0.56    0.59    0.5    0.48    0.65
#FUNNY    0.29    0.38    0.08    0.12    0.32
#HOMO    0.87    0.89    0.9    0.9    0.95
#HETER    0.87    0.87    0.87    0.86    0.93
#TV    0.7    0.68    0.54    0.63    0.81
#BOOK    0.17    0.15    0.06    0.11    0.15
#EVAL    0.72    0.73    0.75    0.73    0.81
#REQ    0.69    0.69    0.67    0.7    0.84
#FACT    0.69    0.69    0.74    0.73    0.82
#REF    0.8    0.79    0.78    0.83    0.93
#QUOTE    0.1    0.1    0.23    0.22    0.66
#ARGUE    0.72    0.72    0.7    0.72    0.78
#SUPPORT    0.46    0.45    0.41    0.41    0.54
#AGAINST    0.53    0.51    0.41    0.43    0.62
#AMAZON    0.91    0.92    0.89    0.86    0.96
#YELP    0.94    0.96    0.94    0.93    0.96
#FUNNY*    0.81    0.81    0.68    0.73    0.82
#BOOK*    0.72    0.7    0.7    0.67    0.74"""


# Accuracy
#data = """Dataset    LR    SVM    CNN    LSTM    BERT
#SUGG,0.777027027,0.77027027,0.744932432,0.755067568,0.861486486
#HOTEL,0.932979429,0.95487724,0.946914399,0.942269409,0.96549436
#SENT,0.869068541,0.922671353,0.894551845,0.898945518,0.90202109
#PARA,0.839421613,0.869101979,0.853120244,0.853120244,0.869101979
#FUNNY-,0.90558868,0.975737803,0.930510695,0.910500895,0.97461281
#HOMO,0.855555556,0.86,0.888888889,0.875555556,0.92
#HETER,0.81741573,0.823033708,0.828651685,0.803370787,0.896067416
#TV,0.667569397,0.658090724,0.665538253,0.648612051,0.742721733
#BOOK-,0.914113574,0.968200397,0.942775715,0.91,0.9592625
#EVAL,0.785085882,0.795978215,0.780896523,0.784666946,0.852953498
#REQ,0.88646837,0.894009217,0.858818601,0.873062421,0.932970256
#FACT,0.781315459,0.818181818,0.810640972,0.806032677,0.871386678
#REF,0.989526602,0.990364474,0.99287809,0.994553833,0.995391705
#QUOTE,0.963133641,0.987431923,0.986594051,0.980310013,0.99078341
#ARGUE,0.703073008,0.727931102,0.722450577,0.731454296,0.788804071
#SUPPORT,0.670972793,0.808377373,0.752201996,0.762967313,0.815032296
#AGAINST,0.680955177,0.769426502,0.736347622,0.766294774,0.800156586
#AMAZON-,0.894551318,0.911326108,0.877901526,0.872251597,0.952900589
#YELP,0.942157895,0.955184211,0.939368421,0.931315789,0.961447368
#FUNNY*,0.791903612,0.80294972,0.748271489,0.726956,0.815079982
#BOOK*,0.713554785,0.701421739,0.698933696,0.69,0.744381061"""


# AUC
data = """Dataset    LR    SVM    CNN    LSTM    BERT
SUGG,0.843989682,0.846512053,0.832667549,0.835235573,0.929750274
HOTEL,0.933139064,0.937242998,0.899991905,0.755091468,0.96838271
SENT,0.859050857,0.860315396,0.8034626,0.850297558,0.915473934
PARA,0.828795752,0.834996156,0.824876391,0.811097715,0.881528593
FUNNY-,0.876396507,0.863728718,0.79,0.64207622,0.866466344
HOMO,0.921278334,0.924137227,0.945757607,0.938712477,0.967658771
HETER,0.868184405,0.889015367,0.89660406,0.8592676,0.956137355
TV,0.727967457,0.730249127,0.730840228,0.703787461,0.823063063
BOOK-,0.742188288,0.699554912,0.67,0.65,0.733362075
EVAL,0.858437675,0.869008869,0.861581387,0.853642442,0.928040946
REQ,0.941446811,0.941994922,0.889681851,0.872395961,0.970358167
FACT,0.871573184,0.885937257,0.880619514,0.88302968,0.941695462
REF,0.99586718,0.997033989,0.991084153,0.994540046,0.999109306
QUOTE,0.943260858,0.921819377,0.881332494,0.921381237,0.972205488
ARGUE,0.812698093,0.811934308,0.803741955,0.814103078,0.878756466
SUPPORT,0.752139796,0.7498748,0.723139355,0.730673894,0.840962338
AGAINST,0.765041736,0.757056379,0.726550871,0.742615807,0.839675757
AMAZON-,0.959394175,0.969422931,0.942824764,0.938451354,0.987827378
YELP,0.986247747,0.991189875,0.794138377,0.977483331,0.993182922
FUNNY*,0.878091689,0.882675787,0.814906659,0.774294325,0.895337897
BOOK*,0.789396637,0.77512214,0.61,0.748828551,0.824213805"""

datasets = {}
for line in data.split("\n")[1:]:
  #items = line.split("    ")
  items = line.split(",")
  datasets[items[0]] = [float(v) for v in items[1:]]

#for ds in datasets:
#  plot(ds, datasets)
#plot_all(datasets)
plot_per_group(group1, datasets, "low.pdf")
plot_per_group(group2, datasets, "high.pdf")
