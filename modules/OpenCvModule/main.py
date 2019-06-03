#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import random
import time
import sys
from cv2 import cv2 as cv
import iothub_client
from iothub_client import IoTHubModuleClient, IoTHubClientError, IoTHubTransportProvider
from iothub_client import IoTHubMessage, IoTHubMessageDispositionResult, IoTHubError, DeviceMethodReturnValue

# HTTP options
# Because it can poll "after 9 seconds" polls will happen effectively
# at ~10 seconds.
# Note that for scalabilty, the default value of minimumPollingTime
# is 25 minutes. For more information, see:
# https://azure.microsoft.com/documentation/articles/iot-hub-devguide/#messaging
TIMEOUT = 241000
MINIMUM_POLLING_TIME = 9

# messageTimeout - the maximum time in milliseconds until a message times out.
# The timeout period starts at IoTHubModuleClient.send_event_to_output.
# By default, messages do not expire.
MESSAGE_TIMEOUT = 10000

RECEIVE_CONTEXT = 0
MESSAGE_COUNT = 5
RECEIVED_COUNT = 0
TWIN_CONTEXT = 0
METHOD_CONTEXT = 0

# global counters
SEND_CALLBACKS = 0

PROTOCOL = IoTHubTransportProvider.MQTT

MSG_TXT = "{\"banana\": %.2f,\"orange\": %.2f,\"other\": %.2f}"


def send_confirmation_callback(message, result, user_context):
    global SEND_CALLBACKS
    print("Confirmation[%d] received for message with result = %s" % (
        user_context, result))
    map_properties = message.properties()
    key_value_pair = map_properties.get_internals()
    print("    Properties: %s" % key_value_pair)
    SEND_CALLBACKS += 1
    print("    Total calls confirmed: %d" % SEND_CALLBACKS)


class HubManager(object):

    def __init__(
            self,
            protocol):
        self.client_protocol = protocol
        self.client = IoTHubModuleClient()
        self.client.create_from_environment(protocol)
        # set the time until a message times out
        self.client.set_option("messageTimeout", MESSAGE_TIMEOUT)
        # set to increase logging level
        # self.client.set_option("logtrace", 1)

    # Sends a message to the queue with outputQueueName, "temperatureOutput" in the case of the sample.

    def send_event_to_output(self, outputQueueName, event, properties, send_context):
        if not isinstance(event, IoTHubMessage):
            event = IoTHubMessage(bytearray(event, 'utf8'))

        if len(properties) > 0:
            prop_map = event.properties()
            for key in properties:
                prop_map.add_or_update(key, properties[key])

        self.client.send_event_async(
            outputQueueName, event, send_confirmation_callback, send_context)


def main(protocol):
    try:
        print("\nPython %s\n" % sys.version)
        print("IoT Hub Client for Python")

        hub_manager = HubManager(protocol)

        print("Starting the IoT Hub Python sample...")

        # content = "Hello World from Python APi"

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        cv_net = cv.dnn.readNetFromTensorflow(
            './model/frozen_inference_graph.pb',
            './model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
        )
        this_dict = {
            "banana": 0,
            "orange": 0,
            "other": 0
        }
        start_time = time.time()
        counter = 0
        while True:
            # Capture frame-by-frame
            ret, img = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            """ rows = img.shape[0]
            cols = img.shape[1] """
            cv_net.setInput(cv.dnn.blobFromImage(
                img, size=(300, 300), swapRB=True, crop=False))

            cv_out = cv_net.forward()

            for detection in cv_out[0, 0, :, :]:
                classId = int(detection[1])
                score = float(detection[2])
                if score > 0.4:
                    """ left = detection[3] * cols
                    top = detection[4] * rows
                    right = detection[5] * cols
                    bottom = detection[6] * rows """
                    if(classId == 52):
                        this_dict["banana"] = this_dict["banana"] + 1
                    elif(classId == 55):
                        this_dict["orange"] = this_dict["orange"] + 1
                    else:
                        this_dict["other"] = this_dict["other"] + 1
            if (time.time() - start_time) > 5:
                msg_txt_formatted = MSG_TXT % (
                    int(this_dict["banana"] / counter),
                    int(this_dict["orange"] / counter),
                    int(this_dict["other"] / counter)
                )
                msg_properties = {}
                hub_manager.send_event_to_output(
                    "detectionOutput", msg_txt_formatted, msg_properties, 1)
                start_time = time.time()
                this_dict = {
                    "banana": 0,
                    "orange": 0,
                    "other": 0
                }
                counter = 0
            counter += 1
    except IoTHubError as iothub_error:
        print("Unexpected error %s from IoTHub" % iothub_error)
        return
    except KeyboardInterrupt:
        print("IoTHubModuleClient sample stopped")


if __name__ == '__main__':
    main(PROTOCOL)
