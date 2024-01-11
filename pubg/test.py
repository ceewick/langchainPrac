import json
from pprint import pprint
import pandas as pd

with open('lifetime.json','r') as file:
    # for x in file:
        # jData = json.loads(x)
    jData = json.load(file)

idNameDict = {
    'SnpCrckPop':"account.faac68b108964de6833b57d91f5bf250",
    'ToucanSamXX':"account.3f86f1e71b70487e866b3d201b83032c",
    'InMyOffice':"account.c29966f76c8c4fa092330d79e5f6714e",
    'UNluckeCharm':"account.533b2072849a483b853c9b2600866150",
    'XCapnCrunchx739':"account.8c7b3f7383d948d587ffb63874c95e23",
    'EatsWheaties':"account.dcd66b303ac4467c83e31efc95b9e739",
    'xHoneySmacks':"account.6654c42c122444c8be1e2f9e459d071c"
    }

# pprint(jData)
# print(type(jData)) ## <class 'dict'>
# print(help(jData))

dataDict = {}
playerId = {}
lifetimeStatsDict = {}
lifetimeStatsNameDict = {}

# pprint(jData['data'])
data = jData['data']
for i in data:
    dataDict = i
    playerDict = dataDict['relationships']['player']['data']
    # playerId['id'] = playerDict.get('id')
    dataDict = dataDict['attributes']['gameModeStats']['squad']
    dataDict['id'] = playerDict.get('id')
    for name,id in idNameDict.items():
        if id in dataDict['id']:
            lifetimeStatsNameDict[name] = dataDict
        else:
            continue
    lifetimeStatsDict.update(lifetimeStatsNameDict)

# pprint(lifetimeStatsDict)

df=pd.DataFrame(lifetimeStatsDict)#['name'])
# df.to_csv('lifetimeStats.csv', sep=',')
pprint(df)