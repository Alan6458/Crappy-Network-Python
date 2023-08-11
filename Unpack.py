import gzip
import numpy as np
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 10000
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
g = gzip.open('t10k-labels-idx1-ubyte.gz','r')
g.read(8)
buf = g.read(10000)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
images = []
answers = []
for i in range(10000):
    images.append("")
    for j in data[i]:
        for k in j:
            images[i] = images[i] + str(int(k[0])).zfill(3)
    images[i] = images[i] + "\n"
    answers.append(str(labels[i]))
    print(i)
file1 = open("imagesTest.txt", "w")
file1.writelines(images)
file1.close()
file2 = open("answersTest.txt", "w")
file2.writelines(answers)
file2.close()