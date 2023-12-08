#encoding=utf-8
from gevent import pywsgi

import view
from view import app
from config import listenIp, listenPort

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=8890, debug=False, processes=1)
    server = pywsgi.WSGIServer((listenIp, listenPort), app)
    server.serve_forever()
