# with open('map_clsloc.txt', 'r') as f:
#     mapper = f.readlines()
# n2name = dict()
# for i in mapper:
#     a, b, c = i.split()
#     n2name[a] = c
# dict1 = {0: 'n01532829', 1: 'n01558993', 2: 'n01704323', 3: 'n01749939', 4: 'n01770081', 5: 'n01843383', 6: 'n01855672', 7: 'n01910747', 8: 'n01930112', 9: 'n01981276', 10: 'n02074367', 11: 'n02089867', 12: 'n02091244', 13: 'n02091831', 14: 'n02099601', 15: 'n02101006', 16: 'n02105505', 17: 'n02108089', 18: 'n02108551', 19: 'n02108915', 20: 'n02110063', 21: 'n02110341', 22: 'n02111277', 23: 'n02113712', 24: 'n02114548', 25: 'n02116738', 26: 'n02120079', 27: 'n02129165', 28: 'n02138441', 29: 'n02165456', 30: 'n02174001', 31: 'n02219486', 32: 'n02443484', 33: 'n02457408', 34: 'n02606052', 35: 'n02687172', 36: 'n02747177', 37: 'n02795169', 38: 'n02823428', 39: 'n02871525', 40: 'n02950826', 41: 'n02966193', 42: 'n02971356', 43: 'n02981792', 44: 'n03017168', 45: 'n03047690', 46: 'n03062245', 47: 'n03075370', 48: 'n03127925', 49: 'n03146219', 50: 'n03207743', 51: 'n03220513', 52: 'n03272010', 53: 'n03337140', 54: 'n03347037', 55: 'n03400231', 56: 'n03417042', 57: 'n03476684', 58: 'n03527444', 59: 'n03535780', 60: 'n03544143', 61: 'n03584254', 62: 'n03676483', 63: 'n03770439', 64: 'n03773504', 65: 'n03775546', 66: 'n03838899', 67: 'n03854065', 68: 'n03888605', 69: 'n03908618', 70: 'n03924679', 71: 'n03980874', 72: 'n03998194', 73: 'n04067472', 74: 'n04146614', 75: 'n04149813', 76: 'n04243546', 77: 'n04251144', 78: 'n04258138', 79: 'n04275548', 80: 'n04296562', 81: 'n04389033', 82: 'n04418357', 83: 'n04435653', 84: 'n04443257', 85: 'n04509417', 86: 'n04515003', 87: 'n04522168', 88: 'n04596742', 89: 'n04604644', 90: 'n04612504', 91: 'n06794110', 92: 'n07584110', 93: 'n07613480', 94: 'n07697537', 95: 'n07747607', 96: 'n09246464', 97: 'n09256479', 98: 'n13054560', 99: 'n13133613'}
#
#
# def seek(x):
#     return n2name[dict1[x]]
#
#
#
# import numpy as np
#
# import matplotlib.pyplot as plt
# import pickle
#
# with open("resnet501d_mini_imgsize64_noDA.txt", "rb") as fp:  # Pickling
#     data64 = pickle.load(fp)
# with open("resnet501d_mini_imgsize32_noDA.txt", "rb") as fp:   # Unpickling
#     data = pickle.load(fp)
# with open("resnet501d_mini_imgsize32_DA.txt", "rb") as fp:   # Unpickling
#     noda_data = pickle.load(fp)
#
# # data = data64
# data64 = np.array(data64)
# data = np.array(data)
# # data = data[[13, -1], :]
#
# data = np.array([data64[-1], data[-1]])
#
# data = data - data[-1, :]
# data = data.tolist()
# tmp = data[0]
# data[0] = sorted(tmp)
#
# c = [i[0] for i in sorted(enumerate(tmp), key=lambda x:x[1])]
#
#
#
# X = np.arange(100)
# fig, ax = plt.subplots()
# for i in range(len(data)):
#     ax.bar(X + i * 0.25, data[i], width=0.25)
# ax.set_xticks(np.arange(len(X)))
# xlabels = []
# for i in c:
#     xlabels.append(seek(i))
# ax.set_xticklabels(xlabels, rotation=90, ha='center', fontsize=9)
# plt.title('ref imgsize32 vs ref imgsize64')
# plt.show()


