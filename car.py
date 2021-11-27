import websocket
import _thread
import time
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
  cannyImage = cv2.Canny(blur, 60, 180)
  return cannyImage


def on_open(ws):
    def run(*args):
        # your car logic here

        cap = cv2.VideoCapture(video_address)

        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        capture = cv2.VideoCapture(video_address)
  
        while(True):

            ret, frame = capture.read()

            grayFrame = CannyEdge(frame)

            cv2.imshow('video gray', grayFrame)
            cv2.imshow('video original', frame)

            if cv2.waitKey(1) == 27:
                break
  
        capture.release()
        cv2.destroyAllWindows()

        while True:
            ret, frame = cap.read()

            gray = CannyEdge(frame)

            cv2.imshow('video gray', gray)
            cv2.imshow('video original', frame)

            if cv2.waitKey(1) == 27:
                break

            # do something based on the frame
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
