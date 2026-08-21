import socket
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

MAX_SPEED = 6.28

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)
server_socket.setblocking(False)

client_socket = None

def move_robot(left, right):
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)

while robot.step(timestep) != -1:
    if client_socket is None:
        try:
            client_socket, addr = server_socket.accept()
            client_socket.setblocking(False)
        except BlockingIOError:
            pass
    else:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                command = data.strip().upper()

                if command == 'A':
                    move_robot(MAX_SPEED, MAX_SPEED)
                elif command == 'B':
                    move_robot(-MAX_SPEED, -MAX_SPEED)
                elif command == 'C':
                    move_robot(MAX_SPEED, -MAX_SPEED)
                elif command == 'D':
                    move_robot(-MAX_SPEED, MAX_SPEED)
                elif command == 'E':
                    move_robot(0.0, 0.0)
        except BlockingIOError:
            pass
        except:
            client_socket.close()
            client_socket = None
