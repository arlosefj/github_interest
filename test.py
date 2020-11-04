# flask api
from flask import Flask, jsonify
import time
import datetime
from flask_restful import reqparse, abort, Api, Resource
import json
import numpy as np

app = Flask(__name__)

api = Api(app)


def listToJson(lst):
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
    return str_json


parser = reqparse.RequestParser()
parser.add_argument('faceimg', type=str)
parser.add_argument('position', type=str)
parser.add_argument('feature', type=str)
parser.add_argument('receivetime', type=str)

def process(faceimg, position, feature, receivetime):
    print(position)
    print(receivetime)
    return 'OK'

class receiveFeature(Resource):
    def post(self):
        args = parser.parse_args()
        faceimg = args['faceimg']
        position = args['position']
        feature = args['feature']
        receivetime = args['receivetime']

        res = process(faceimg, position, feature, receivetime)
        json_data = jsonify({'results': res})
        return json_data

class receiveFeatureGet(Resource):
    def get(self, stringinfo):
        res = stringinfo
        json_data = jsonify({'results': res})
        return json_data



api.add_resource(receiveFeature, '/receiveFeature') 
api.add_resource(receiveFeatureGet, '/receiveFeatureGet/<stringinfo>') 

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True, port=8066)