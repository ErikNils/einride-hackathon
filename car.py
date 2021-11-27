from numpy.core.arrayprint import str_format
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


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return line_image


def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage

def region_of_interest(image):
    height = image.shape[0]
    rectangle = np.array([[(0, height),(0, 50), (160,50), (160, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    print("Left line: " + str(left_line))
    print("Right line: " + str(right_line))
    return np.array((left_line, right_line))

def line_intersection(lines):
    line1 = ([lines[0][0][0],lines[0][0][1]],[lines[0][0][2],lines[0][0][3]])
    line2 = ([lines[1][0][0],lines[1][0][1]],[lines[1][0][2],lines[1][0][3]])
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y




def on_open(ws):
    def run(*args):
        # your car logic here

        cap = cv2.VideoCapture(video_address)

        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        capture = cv2.VideoCapture(video_address)
  
        
        throttle = 0.2
        while True:
            ret, frame = cap.read()

            gray = CannyEdge(frame)

            cropped_video = region_of_interest(gray)
            rho = 2
            theta = np.pi/180
            threshold = 50
            lines = cv2.HoughLinesP(cropped_video,rho, theta, threshold, np.array ([]), minLineLength=5, maxLineGap=5)
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            #avg_image = display_lines(frame, averaged_lines[1]-averaged_lines[0])
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            #print(averaged_lines)
            #print("Average: " + str(averaged_lines[1]-averaged_lines[0]))
            print("Intersection: " + str(averaged_lines))
            
            cv2.imshow('Lane Lines', line_image)
            #cv2.imshow("Avg video", avg_image)
            cv2.imshow("Wombo Combo", combo_image)
            cv2.imshow("Cropped video", cropped_video)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            opposite, adjacent = line_intersection(averaged_lines)

            opposite -= 80

            arctan = np.arctan(opposite/adjacent)
            print("Arctan: " + str(arctan))

            if arctan > 0: angle = 0.2
            else: angle = -0.2

            # do something based on the frame
            #angle = 0.0
            

            message = f"{{\"angle\":{angle},\"throttle\":{throttle},\"drive_mode\":\"user\",\"recording\":false}}"
            ws.send(message)
            print(message)

        #angle = 0.0
        #throttle = 0.0
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
