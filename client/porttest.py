import requests
import sys

serverip = "39.101.161.66"
#serverip = "localhost"
port = "80"
# port = "5000"
router = "/"
# router = ''
url = "http://" + serverip + ":" + port + "/" + router

def main(argv):
    ret = requests.get(url)
    print(ret.text)


if __name__ == '__main__':
    main(sys.argv)
