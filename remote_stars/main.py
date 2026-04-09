import socket

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

host = "84.237.21.36"
port = 5152

# def find_center(image, label) -> tuple[int, int]:
#     y, x = np.where(image == label)
#     return (int(np.mean(y)), int(np.mean(x)))

def recvall(sock, nbytes) -> None | bytearray:
    data = bytearray()
    while len(data) < nbytes:
        package = sock.recv(nbytes - len(data))
        if not package:
            return None
        data.extend(package)
    return data

def solve(image) -> float:
    labeled = label(image > 0)
    center1 = np.unravel_index(np.argmax(image * (labeled == 1)), image.shape)
    center2 = np.unravel_index(np.argmax(image * (labeled == 2)), image.shape)
    return np.linalg.norm(np.array(center1) - np.array(center2))

def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.send(b"124ras1")
        print(sock.recv(10))

        beat = b"nope"
        count = 0
        while beat != b"yep":
            sock.send(b"get")
            bts = recvall(sock, 40002)
            print("image received" if bts else "oops...")
            image = np.frombuffer(bts[2:40002], dtype="uint8").reshape(bts[0], bts[1])
            answer = solve(image)
            sock.send(("%.1f" % round(answer, 1)).encode())
            print(sock.recv(10))
            sock.send(b"beat")
            beat = sock.recv(10)
            count += 1
        print(count)

if __name__ == "__main__":
    main()