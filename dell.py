with open('/data/users/yuefan/fanyue/dconv/checkpoints/dell1/log.txt', 'r') as f:
    l = f.readlines()
    l = l[1:]
    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []
    for ll in l:
        tmp = ll.split()
        # print(tmp)
        train_loss.append(float(tmp[1]))
        test_loss.append(float(tmp[2]))
        train_acc.append(float(tmp[3]))
        test_acc.append(float(tmp[4]))
print(train_loss)
import matplotlib.pyplot as plt
x = range(len(train_loss))


plt.plot(x,train_loss, label="train_loss")
plt.plot(x,test_loss, label="test_loss")
# plt.plot(x,train_acc, label="train_acc")
# plt.plot(x,test_acc, label="test_acc")
plt.legend(loc='lower right')
plt.title("VGG16")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()