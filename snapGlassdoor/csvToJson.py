import csv
import json

def make_json(csvFilePath, jsonFilePath ):
    data = {}
    with open(csvFilePath, 'r', encoding='utf-8') as snap_csv:
        csvReader = csv.DictReader(snap_csv)
        for rows in csvReader:
            key = rows['key']
            data[key] = rows

    with open(jsonFilePath, 'w', encoding='utf-8') as snapJsonFile:
        snapJsonFile.write(json.dumps(data, indent=4))

csvFilePath = 'snapGlassdoor2017.csv'
jsonFilePath = 'snapGlassdoor2017.json'

make_json(csvFilePath, jsonFilePath)