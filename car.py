import websocket
import _thread
import time
import numpy as np
import cv2

host = "192.168.43.243"
port = 8887

socket_address = f"ws://{host}:{port}/wsDrive"
video_address = f"http://{host}:{port}/video"

def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")





def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage

def region_of_interest(image):
    height = image.shape[0]
    rectangle = np.array([[(0, height),(0, 40), (160,40), (160, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image




def on_open(ws):
    def run(*args):
        # your car logic here

        cap = cv2.VideoCapture(video_address)

        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        capture = cv2.VideoCapture(video_address)
  
        

        while True:
            ret, frame = cap.read()

            gray = CannyEdge(frame)

            cropped_video = region_of_interest(gray)
            

            cv2.imshow('video gray', gray)
            cv2.imshow('video original', frame)
            cv2.imshow("Final video", cropped_video)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # do something based on the frame
            angle = 0.0
            throttle = 0.0

            message = f"{{\"angle\":{angle},\"throttle\":{throttle},\"drive_mode\":\"user\",\"recording\":false}}"
            ws.send(message)
            print(message)

        angle = 0.0
        throttle = 0.0
        message = f"{{\"angle\":{angle},\"throttle\":{throttle},\"drive_mode\":\"user\",\"recording\":false}}"
        ws.send(message)
        print(message)
  
        capture.release()
        cv2.destroyAllWindows()

    _thread.start_new_thread(run, ())


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket_address,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()
