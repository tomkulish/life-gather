import sys, traceback
import pymongo
import paho.mqtt.client as mqtt
import json

__author__ = 'tbk'

passwd = ""
ip = ""
port = ""
caCert = ""

def post_data_in_mongo(data):
    print "Trying to insert the data into mongo"
    try:
    # establish a connection to the database
    	connection = pymongo.MongoClient("mongodb://localhost")

    # get a handle to the school database
    	db = connection.lifegather
    	location = db.location
	
        location.insert_one(data)
        print "Inserting the data"
    except:
        print "Error inserting data"
        print "Unexpected error:", sys.exc_info()[0]
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60


# The callback for when the client successfully connects to the broker
def on_connect(client, userdata, rc):
    ''' We subscribe on_connect() so that if we lose the connection
        and reconnect, subscriptions will be renewed. '''

    client.subscribe("owntracks/+/+")

# The callback for when a PUBLISH message is received from the broker
def on_message(client, userdata, msg):

    topic = msg.topic

    try:
        data = json.loads(str(msg.payload))

	print data
        print "TID = {0} is currently at {1}, {2}".format(data['tid'], data['lat'], data['lon'])
	post_data_in_mongo(data)
    except:
        print "Cannot decode data on topic {0}".format(topic)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.tls_set("/home/tkulish/installingMosquitto/myCA/ca.crt")
client.username_pw_set("owntracks", passwd)

client.connect(ip, port, 60)

# Blocking call which processes all network traffic and dispatches
# callbacks (see on_*() above). It also handles reconnecting.

client.loop_forever()
