from flask import Flask, request
import read_data
import numpy as np
import json
from flask_cors import CORS

# app是Flask的实例，它接收包或者模块的名字作为参数，但一般都是传递__name__
APP = Flask(__name__)
CORS(APP)

# 使用app.route装饰器会将URL和执行的视图函数的关系保存到app.url_map属性上，利用路由进行处理的
@APP.route("/<name>") # /<name>为路由参数
def index(name):
    return "hello" + name

@APP.route("/aqi", methods=['GET','POST'])
def get_aqi_value():
    """
    根据用户提供的输入数据，完成aqi值的预测
    """
    json_data = request.get_data()
    json_data = json.loads(json_data.decode('utf-8'))
    # PM2.5,PM10,CO,No2,So2,O3
    pm25 = json_data.get('pm25')
    pm10 = json_data.get('pm10')
    co = json_data.get('co')
    no2 = json_data.get('no2')
    so2 = json_data.get('so2')
    o3 = json_data.get('o3')
    input_data = [pm25, pm10, co, no2, so2, o3]

    x = np.array(input_data)
    x = read_data.standard_data(x)
    # 从文件中读取theta
    with open('model.txt', 'r') as f:
        theta = np.array([float(line) for line in f.readlines()]).reshape(6, 1)
    aqi_value = np.dot(x, theta)
    return json.dumps({'result':aqi_value[0]})

# 使用这个判断可以保证当其他文件引用这个文件的时候不会执行这个判断内的代码，也就是不会执行app.run函数。
if __name__ ==  "__main__":
# 执行app.run就可以启动服务了。默认Flask只监听虚拟机的本地127.0.0.1这个地址，端口为5000。
    APP.run()
