from view import app
from methods.minic import Minic
from methods.executor import Executor

exe = Executor()

@app.route('/', methods=['GET','POST'])
def index():
    return 'Service is running...'

@app.route('/receive', methods=['GET','POST'])
def receive():
    return Minic.receive()

@app.route('/base64Decode', methods=['GET','POST'])
def base64Decode():
    return Minic.base64Decode()

@app.route('/dumpImage', methods=['POST'])
def dumpImage():
    return Minic.dump()

@app.route('/readImageOpenCV', methods=['POST'])
def readImageOpenCV():
    return Minic.readByOpenCV()

@app.route('/readImagePIL', methods=['POST'])
def readImagePIL():
    return Minic.readByPIL()

@app.route('/getDetResult', methods=['GET'])
def getDetResult():
    result = exe.getOneResult()
    return result.to_json()